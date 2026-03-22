#!/usr/bin/env python3
"""
Federated client: TinyLlama-1.1B training contribution.
Connects to federated server, trains LoRA layers, sends gradients.

This PC (8GB RAM, CPU-only) contributes to training a 1.1B param model
by training locally and sharing gradients with the federation.
"""
import os, sys, time, torch, logging, socket, json, struct
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fed-client")

DEVICE = "cpu"
DTYPE = torch.float32
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_RANK = 4
LORA_ALPHA = 8
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
CLIENT_ID = f"pc-{socket.gethostname()}"


def find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FederatedClient:
    """Connect to federated server, train locally, exchange gradients."""

    def __init__(self, server_host: str, server_port: int, model_id: str = MODEL_ID):
        self.server_host = server_host
        self.server_port = server_port
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.lora_count = 0
        self.round_num = 0
        self.sock = None

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(30)
            self.sock.connect((self.server_host, self.server_port))
            log.info(f"Connected to server at {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            log.warning(f"Could not connect to server: {e}")
            return False

    def send_json(self, data: dict):
        msg = json.dumps(data).encode("utf-8")
        header = struct.pack("!I", len(msg))
        self.sock.sendall(header + msg)

    def recv_json(self) -> dict:
        header = self.sock.recv(4)
        if len(header) < 4:
            return {}
        size = struct.unpack("!I", header)[0]
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                break
            data += chunk
        return json.loads(data.decode("utf-8"))

    def send_gradients(self, gradients: dict):
        """Send gradient dict to server."""
        self.send_json({"type": "gradients", "client_id": CLIENT_ID, "round": self.round_num})
        # Send each tensor
        for name, tensor in gradients.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            data = tensor.cpu().numpy().tobytes()
            header = struct.pack("!I", len(data))
            name_bytes = name.encode("utf-8")
            name_len = struct.pack("!I", len(name_bytes))
            self.sock.sendall(name_len + name_bytes + header + data)

    def recv_model_update(self) -> dict:
        """Receive aggregated model update from server."""
        grads = {}
        n_tensors = self.recv_json().get("n_tensors", 0)
        for _ in range(n_tensors):
            name_len_data = self.sock.recv(4)
            if len(name_len_data) < 4:
                break
            name_len = struct.unpack("!I", name_len_data)[0]
            name = self.sock.recv(name_len).decode("utf-8")
            size_data = self.sock.recv(4)
            if len(size_data) < 4:
                break
            size = struct.unpack("!I", size_data)[0]
            data = b""
            while len(data) < size:
                chunk = self.sock.recv(min(65536, size - len(data)))
                data += chunk
            import numpy as np
            grads[name] = torch.from_numpy(np.frombuffer(data, dtype=np.float32))
        return grads

    def load_model(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        try:
            log.info(f"Loading tokenizer: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            log.info(f"Loading model: {self.model_id}")
            t0 = time.time()
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, config=config, trust_remote_code=True, torch_dtype=DTYPE,
            )
            n_params = sum(p.numel() for p in self.model.parameters())
            log.info(f"  Loaded in {time.time()-t0:.1f}s | {n_params/1e6:.1f}M params")
            return True
        except Exception as e:
            log.error(f"Model load failed: {e}")
            return False

    def apply_lora(self) -> int:
        from lisa.train_torch import LoRALinear
        import torch.nn as nn
        count = 0
        for full_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(tm in full_name.lower() for tm in ["attn", "mlp", "fc", "proj"]):
                continue
            lora = LoRALinear(module, rank=LORA_RANK, alpha=LORA_ALPHA,
                              dropout=0.05, target_module_name=full_name)
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                try:
                    parent = self.model.get_submodule(parts[0])
                    setattr(parent, parts[1], lora)
                    count += 1
                except (KeyError, AttributeError):
                    pass
        log.info(f"  LoRA applied to {count} layers")
        self.lora_count = count
        return count

    def train_local(self, texts, n_steps: int = 5) -> float:
        """
        Train LoRA layers locally on provided texts. Returns avg loss.
        NOTE: Does NOT call optimizer.step() — caller extracts gradients
        for sending to federated server.
        """
        if texts is None or len(texts) == 0:
            log.info("No data from server, using synthetic batch")
            texts = ["The quick brown fox jumps over the lazy dog."] * 100

        # Tokenize
        enc = self.tokenizer(texts, truncation=True, max_length=128,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].clamp(0, self.tokenizer.vocab_size - 1)
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        # Freeze ALL params, then unfreeze only LoRA params
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"  Training {trainable:,} LoRA params for {n_steps} steps (no optimizer.step)")

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=0.01,
        )

        self.model.train()
        losses = []
        batch_size = 4

        for step in range(n_steps):
            idx = torch.randperm(len(input_ids))[:batch_size].tolist()
            ids = input_ids[idx]
            mask = attention_mask[idx]
            labs = labels[idx]

            optimizer.zero_grad()
            outputs = self.model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labs.view(-1),
                ignore_index=self.tokenizer.pad_token_id or -100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            # DO NOT call optimizer.step() here - we need gradients for the server
            losses.append(loss.item())

            if (step + 1) % 5 == 0:
                log.info(f"    Step {step+1}/{n_steps} | loss={loss.item():.4f}")

        avg_loss = sum(losses) / len(losses)
        log.info(f"  Local training done: avg_loss={avg_loss:.4f} (gradients ready for server)")
        return avg_loss

    def apply_gradients(self, updates: dict):
        """Apply aggregated gradient updates from server (accumulate to LoRA params)."""
        for name, tensor in updates.items():
            if name in dict(self.model.named_parameters()):
                param = dict(self.model.named_parameters())[name]
                # Accumulate gradient update rather than replacing the param
                update = tensor.to(param.device)
                if update.shape == param.shape:
                    param.data.add_(update)
        log.info(f"  Applied {len(updates)} gradient updates (accumulated)")

    def run_federated_round(self, data: list = None) -> bool:
        """One round: train locally, send gradients, receive update."""
        log.info(f"\n=== Federated Round {self.round_num} ===")

        # Train locally
        log.info("Training locally...")
        avg_loss = self.train_local(data, n_steps=5)

        # Extract gradients BEFORE optimizer.step() clears them
        grads = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.grad is not None:
                grads[name] = param.grad.clone().cpu()
        log.info(f"  Extracted {len(grads)} gradient tensors")

        if self.sock is None:
            log.info("No server connection - saving gradients locally")
            out_dir = Path("output/federated_grads")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(grads, out_dir / f"round_{self.round_num}_grads.pt")
            self.round_num += 1
            return True

        try:
            # Send gradients
            self.send_gradients(grads)

            # Receive aggregated update
            log.info("Waiting for model update from server...")
            updates = self.recv_model_update()
            if updates:
                self.apply_gradients(updates)
                log.info("  Round complete - model updated")
            else:
                log.info("  No updates received from server")

            self.round_num += 1
            return True

        except Exception as e:
            log.warning(f"Server communication error: {e}")
            # Save locally as fallback
            out_dir = Path("output/federated_grads")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(grads, out_dir / f"round_{self.round_num}_grads.pt")
            self.round_num += 1
            return True

    def disconnect(self):
        if self.sock:
            try:
                self.send_json({"type": "disconnect", "client_id": CLIENT_ID})
            except:
                pass
            self.sock.close()
            self.sock = None


def run_standalone():
    """Run without server - train locally, demonstrate the concept."""
    log.info("=" * 60)
    log.info("FEDERATED LEARNING CLIENT (Standalone Demo)")
    log.info("=" * 60)
    log.info(f"Client ID: {CLIENT_ID}")
    log.info("No server - demonstrating local training as if in federation")
    log.info("In a real deployment, this client would connect to:")
    log.info("  python -m federated.server --model TinyLlama-1.1B --port 8080")
    log.info("")

    client = FederatedClient(SERVER_HOST, SERVER_PORT)

    if not client.load_model():
        return
    client.apply_lora()
    # LoRA params are applied but frozen; train_local() unfreezes them per round
    log.info("LoRA applied. train_local() will unfreeze LoRA params per round.")

    # Try to connect to server
    connected = client.connect()

    # Run multiple federated rounds
    for round_i in range(5):
        client.round_num = round_i
        client.run_federated_round(data=None)
        time.sleep(0.5)

    if connected:
        client.disconnect()
    else:
        log.info("\nStandalone demo complete. Gradients saved to output/federated_grads/")
        log.info("To run as part of a federation, start a server first.")


if __name__ == "__main__":
    run_standalone()
