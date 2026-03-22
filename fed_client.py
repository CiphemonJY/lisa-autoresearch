#!/usr/bin/env python3
"""
Federated client: TinyLlama-1.1B training contribution.
Connects to federated server, trains LoRA layers, sends gradients.

This PC (8GB RAM, CPU-only) contributes to training a 1.1B param model
by training locally and sharing gradients with the federation.
"""
import os, sys, time, torch, logging, socket, json, struct, pickle, argparse
from pathlib import Path
from typing import Optional

from utils.audit_logger import AuditLogger

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

    def __init__(self, server_host: str, server_port: int, model_id: str = MODEL_ID, auth_token: Optional[str] = None):
        self.server_host = server_host
        self.server_port = server_port
        self.model_id = model_id
        self.auth_token = auth_token
        self.model = None
        self.tokenizer = None
        self.lora_count = 0
        self.round_num = 0
        self.sock = None
        self.audit_logger = AuditLogger(audit_dir="audit_logs")

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(30)
            self.sock.connect((self.server_host, self.server_port))
            log.info(f"Connected to server at {self.server_host}:{self.server_port}")
            self.audit_logger.log_event(
                event_type="client_connect",
                client_id=CLIENT_ID,
                ip_address=f"{self.server_host}:{self.server_port}",
                success=True,
            )

            # Send auth token first if configured
            if self.auth_token is not None:
                token_bytes = self.auth_token.encode("utf-8")
                self.sock.sendall(struct.pack("!I", len(token_bytes)) + token_bytes)
                log.info("Auth token sent to server")
            else:
                # Send zero-length token (server will skip auth if it has no token set)
                self.sock.sendall(struct.pack("!I", 0))

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
        """Send gradient dict to server using torch serialization (preserves shapes).

        Uses chunked streaming for tensors > 10MB to avoid OOM.
        Protocol: JSON header + per-tensor frames.
        Each frame: [name_len(4)][name][dtype_len(4)][dtype][shape_len(4)][shape_json]
                   [num_bytes(8)][chunk1][chunk2]...
        Small tensors (< 10MB) use simple pickle for efficiency.
        """
        use_streaming = True  # Always attempt to detect large tensors
        STREAMING_THRESHOLD = 10 * 1024 * 1024  # 10 MB

        self.send_json({"type": "gradients", "client_id": CLIENT_ID, "round": self.round_num})

        # Detect if any tensor exceeds threshold
        large_tensors = {}
        small_tensors = {}
        for name, tensor in gradients.items():
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensor_bytes > STREAMING_THRESHOLD:
                large_tensors[name] = tensor
            else:
                small_tensors[name] = tensor

        # Send header: number of large tensors
        header = {
            "n_large": len(large_tensors),
            "n_small": len(small_tensors),
            "large_names": list(large_tensors.keys()),
        }
        self.send_json({"grad_header": header, "type": "gradients", "client_id": CLIENT_ID, "round": self.round_num})

        # Send small tensors as pickle blob
        if small_tensors:
            small_data = pickle.dumps(small_tensors)
            self.sock.sendall(struct.pack("!I", len(small_data)) + small_data)
        else:
            self.sock.sendall(struct.pack("!I", 0))

        # Send large tensors using chunked streaming
        for name, tensor in large_tensors.items():
            self._send_tensor_streaming(name, tensor)

    def _send_tensor_streaming(self, name: str, tensor: torch.Tensor, chunk_size: int = 65536):
        """Send a tensor in chunks to avoid memory spike."""
        name_bytes = name.encode("utf-8")
        self.sock.sendall(struct.pack("!I", len(name_bytes)) + name_bytes)

        dtype_str = str(tensor.dtype).replace("torch.", "")
        dtype_bytes = dtype_str.encode("utf-8")
        self.sock.sendall(struct.pack("!I", len(dtype_bytes)) + dtype_bytes)

        shape_bytes = json.dumps(list(tensor.shape)).encode("utf-8")
        self.sock.sendall(struct.pack("!I", len(shape_bytes)) + shape_bytes)

        total_bytes = tensor.numel() * tensor.element_size()
        self.sock.sendall(struct.pack("!Q", total_bytes))  # 8-byte unsigned

        np_bytes = tensor.cpu().numpy().tobytes()
        for offset in range(0, len(np_bytes), chunk_size):
            chunk = np_bytes[offset : offset + chunk_size]
            self.sock.sendall(chunk)

    def _recv_tensor_streaming(self, chunk_size: int = 65536) -> Tuple[str, torch.Tensor]:
        """Receive a streamed tensor: [name_len][name][dtype_len][dtype][shape_len][shape][num_bytes][chunks]."""
        name_len_data = self._recv_exact(4)
        if len(name_len_data) < 4:
            return "", torch.zeros(0)
        name_len = struct.unpack("!I", name_len_data)[0]
        name_bytes = self._recv_exact(name_len)
        name = name_bytes.decode("utf-8")

        dtype_len_data = self._recv_exact(4)
        if len(dtype_len_data) < 4:
            return name, torch.zeros(0)
        dtype_len = struct.unpack("!I", dtype_len_data)[0]
        dtype_bytes = self._recv_exact(dtype_len)
        dtype_str = dtype_bytes.decode("utf-8")

        shape_len_data = self._recv_exact(4)
        if len(shape_len_data) < 4:
            return name, torch.zeros(0)
        shape_len = struct.unpack("!I", shape_len_data)[0]
        shape_bytes = self._recv_exact(shape_len)
        shape = json.loads(shape_bytes.decode("utf-8"))

        num_bytes_data = self._recv_exact(8)
        if len(num_bytes_data) < 8:
            return name, torch.zeros(0)
        total_bytes = struct.unpack("!Q", num_bytes_data)[0]

        buffer = b""
        while len(buffer) < total_bytes:
            chunk = self.sock.recv(min(chunk_size, total_bytes - len(buffer)))
            if not chunk:
                break
            buffer += chunk

        arr = np.frombuffer(buffer, dtype=dtype_str).copy()
        result = torch.from_numpy(arr).view(shape)
        return name, result

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return data
            data += chunk
        return data

    def recv_model_update(self) -> dict:
        """Receive aggregated model update from server (pickle-serialized dict).

        Handles compressed payloads if the server sends compression metadata.
        Handles streaming large tensors (>10MB) sent chunk-by-chunk.
        On socket timeout, returns empty dict (callers can reconnect).
        """
        grads = {}
        try:
            # Receive JSON header + body
            header = self.sock.recv(4)
            if len(header) < 4:
                log.warning("No response header from server")
                return grads
            meta_len = struct.unpack("!I", header)[0]
            meta_bytes = b""
            while len(meta_bytes) < meta_len:
                chunk = self.sock.recv(meta_len - len(meta_bytes))
                if not chunk:
                    break
                meta_bytes += chunk
            try:
                meta = json.loads(meta_bytes.decode("utf-8"))
                log.info(f"  Server response: {meta}")
            except Exception as e:
                log.warning(f"Failed to parse server response JSON: {e}")
                return grads

            # Check for error / round-failed response
            msg_type = meta.get("type", "")
            if msg_type == "error" or msg_type == "round_failed":
                log.warning(f"Server reported round failure: {meta.get('message', 'unknown')}")
                return grads

            compression_meta = meta.get("compression", {})
            compression_method = compression_meta.get("method", "none")
            use_streaming = meta.get("use_streaming", False)
            large_names = set(meta.get("large_tensors", []))

            # Receive small tensors as pickle blob
            n_header = self.sock.recv(4)
            if len(n_header) < 4:
                return grads
            n_bytes = struct.unpack("!I", n_header)[0]
            raw = b""
            while len(raw) < n_bytes:
                chunk = self.sock.recv(min(65536, n_bytes - len(raw)))
                if not chunk:
                    break
                raw += chunk

            if raw and n_bytes > 0:
                small_tensors = pickle.loads(raw)
                grads.update(small_tensors)

            # Receive large tensors via streaming
            if use_streaming and large_names:
                for name in large_names:
                    _, tensor = self._recv_tensor_streaming()
                    grads[name] = tensor
                log.info(f"  Received {len(large_names)} streamed large tensors")

            self.audit_logger.log_event(
                event_type="model_update_receive",
                client_id=CLIENT_ID,
                data_type="aggregated_model",
                record_count=len(grads),
                success=True,
                epoch=str(self.round_num),
            )
        except socket.timeout:
            log.warning("Timeout waiting for model update from server")
            return grads
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            log.warning(f"Server disconnected while sending update: {e}")
            return grads
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
        """Apply LoRA using the LoraAppliedModel from train_torch (handles Conv1D too)."""
        from lisa.train_torch import LoraAppliedModel, LISAConfig
        
        # Create a minimal config for LoRA application
        class _Cfg:
            lora_rank = LORA_RANK
            lora_alpha = LORA_ALPHA
            lora_dropout = 0.05
            lora_target_modules = ["c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]
        
        cfg = _Cfg()
        wrapper = LoraAppliedModel(self.model, cfg)
        count = wrapper.apply_lora(target_modules=cfg.lora_target_modules)
        wrapper.freeze_all_except_lora()
        self.lora_count = count
        log.info(f"  LoRA applied to {count} layers")
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

        self.audit_logger.log_event(
            event_type="data_access",
            client_id=CLIENT_ID,
            data_type="training_data",
            record_count=len(texts),
            success=True,
            epoch=str(self.round_num),
        )

        # Ensure LoRA is applied (handles both first-time and reconnect)
        if self.lora_count == 0:
            self.apply_lora()

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
        """One round: train locally, send gradients, receive update.

        On socket timeout receiving the update, tries to reconnect up to 3 times
        before saving gradients locally as fallback.
        """
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
        self.audit_logger.log_event(
            event_type="gradient_send",
            client_id=CLIENT_ID,
            data_type="gradient_update",
            record_count=len(grads),
            success=True,
            epoch=str(self.round_num),
        )

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

            # Receive aggregated update (with timeout-based reconnection)
            log.info("Waiting for model update from server...")
            max_retries = 3
            updates = {}
            for attempt in range(1, max_retries + 1):
                updates = self.recv_model_update()
                if updates:
                    break
                if attempt < max_retries:
                    log.warning(
                        f"Timeout/disconnect receiving update — attempt {attempt}/{max_retries}, "
                        f"reconnecting..."
                    )
                    self.sock.close()
                    self.sock = None
                    if self.connect():
                        # Resend gradients on fresh connection
                        self.send_gradients(grads)
                    else:
                        log.warning("Reconnect failed")
                        break
            else:
                log.warning(
                    f"Failed to receive update after {max_retries} attempts — "
                    f"saving gradients locally"
                )

            if updates:
                self.apply_gradients(updates)
                log.info("  Round complete - model updated")
            else:
                log.info("  No updates received from server — saving locally as fallback")
                out_dir = Path("output/federated_grads")
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(grads, out_dir / f"round_{self.round_num}_grads.pt")

            self.round_num += 1
            return True

        except socket.timeout:
            log.warning(f"Socket timeout in round {self.round_num} — saving gradients locally")
            out_dir = Path("output/federated_grads")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(grads, out_dir / f"round_{self.round_num}_grads.pt")
            self.round_num += 1
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            log.warning(f"Server disconnected during round {self.round_num}: {e}")
            out_dir = Path("output/federated_grads")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(grads, out_dir / f"round_{self.round_num}_grads.pt")
            self.round_num += 1
            return True

    def disconnect(self):
        if self.sock:
            try:
                self.send_json({"type": "disconnect", "client_id": CLIENT_ID})
            except Exception as e:
                log.warning(f"Failed to send disconnect message to server: {e}")
            self.audit_logger.log_event(
                event_type="client_disconnect",
                client_id=CLIENT_ID,
                success=True,
            )
            self.sock.close()
            self.sock = None


def run_standalone():
    """Run without server - train locally, demonstrate the concept."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--auth-token", type=str, default=None,
                        help="Auth token for server (optional)")
    args, unknown = parser.parse_known_args()

    log.info("=" * 60)
    log.info("FEDERATED LEARNING CLIENT (Standalone Demo)")
    log.info("=" * 60)
    log.info(f"Client ID: {CLIENT_ID}")
    log.info("No server - demonstrating local training as if in federation")
    log.info("In a real deployment, this client would connect to:")
    log.info("  python -m federated.server --model TinyLlama-1.1B --port 8080")
    log.info("")

    client = FederatedClient(args.host, args.port, args.model, auth_token=args.auth_token)

    if not client.load_model():
        return
    client.apply_lora()
    log.info("LoRA applied. Starting federated rounds.")

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


def main():
    """Entry point for lisa-client CLI."""
    run_standalone()

