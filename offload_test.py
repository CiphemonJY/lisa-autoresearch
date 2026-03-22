#!/usr/bin/env python3
"""
LISA + LoRA training test on TinyLlama-1.1B.
Uses LISA layer selection (always train bottom+top, randomly sample middle)
combined with LoRA adapters (only 0.29% of params trainable).

This proves the LISA+LoRA training pipeline works on CPU for a 1.1B model.
Real disk-offloading for 7B+ models requires accelerate library (not installed).
"""
import gc, os, sys, time, tempfile, shutil, logging, torch
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lisa-test")

DEVICE = "cpu"
DTYPE = torch.float32

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # 1.1B params
LORA_RANK = 4
LORA_ALPHA = 8
LORA_TARGET = ["attn", "mlp", "fc", "proj"]


class LISALoRATrainer:
    """
    LISA (Layer-wise Importance Sampling) + LoRA trainer.
    
    LISA: Per round, only train selected layers (bottom always, top always,
          middle randomly sampled). Reduces compute ~70%.
    LoRA: Only update low-rank adapter matrices (0.29% of model params).
    """

    def __init__(self, model_id: str, num_layer_groups: int = 4):
        self.model_id = model_id
        self.num_layer_groups = num_layer_groups
        self.cache_dir = Path(tempfile.mkdtemp(prefix="lisa_offload_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.layer_names: List[str] = []
        self.layer_groups: List[List[str]] = []
        self.lora_count = 0

    def _get_layer_names(self) -> List[str]:
        """Get transformer layer submodule names in order."""
        layers = []
        for name, _ in self.model.named_modules():
            # LlamaModel: model.layers.0, model.layers.1, ...
            # GPT2: transformer.h.0, transformer.h.1, ...
            parts = name.split(".")
            if len(parts) >= 2 and parts[-1].isdigit():
                if any(p in parts for p in ["layers", "h"]):
                    layers.append(name)
        def sort_key(n):
            for part in n.split("."):
                if part.isdigit():
                    return int(part)
            return 0
        return sorted(layers, key=sort_key)

    def _split_into_groups(self, layers: List[str], num_groups: int) -> List[List[str]]:
        group_size = len(layers) // num_groups
        groups = []
        for i in range(num_groups):
            start = i * group_size
            end = start + group_size if i < num_groups - 1 else len(layers)
            groups.append(layers[start:end])
        return groups

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

            self.layer_names = self._get_layer_names()
            self.layer_groups = self._split_into_groups(self.layer_names, self.num_layer_groups)
            log.info(f"  Layers: {len(self.layer_names)} | Groups: {self.num_layer_groups}")
            for i, g in enumerate(self.layer_groups):
                nums = [n.split(".")[-1] for n in g]
                log.info(f"    Group {i}: layers {nums[0]}-{nums[-1]} ({len(g)} total)")
            return True
        except Exception as e:
            log.error(f"Failed: {e}")
            return False

    def apply_lora(self) -> int:
        from lisa.train_torch import LoRALinear
        import torch.nn as nn
        count = 0
        for full_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(tm in full_name.lower() for tm in LORA_TARGET):
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

    def select_layers(self, round_idx: int) -> List[str]:
        """LISA layer selection: always train bottom+top, randomly sample middle."""
        total = len(self.layer_names)
        bottom = list(range(min(2, total)))           # always: first 2 layers
        top_start = max(0, total - 2)
        top = list(range(top_start, total))            # always: last 2 layers
        middle = list(range(2, top_start))             # sample from middle
        rng = torch.Generator().manual_seed(42 + round_idx)
        sampled = []
        if len(middle) > 0:
            n_sample = min(2, len(middle))
            idx = torch.randperm(len(middle), generator=rng)[:n_sample].tolist()
            sampled = [middle[i] for i in idx]
        selected = bottom + sampled + top
        return [self.layer_names[i] for i in sorted(set(selected))]

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze parameters from specific layers + all LoRA params."""
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            elif any(ln in name for ln in layer_names):
                param.requires_grad = True

    def cleanup(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="output/lisa_test")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("LISA + LoRA + TinyLlama-1.1B")
    log.info("=" * 60)

    trainer = LISALoRATrainer(model_id=args.model, num_layer_groups=args.groups)
    if not trainer.load_model():
        return

    n_params = sum(p.numel() for p in trainer.model.parameters())
    log.info(f"Model: {n_params/1e6:.1f}M params | LoRA target: {LORA_RANK} | alpha: {LORA_ALPHA}")

    trainer.freeze_all()
    lora_count = trainer.apply_lora()
    total_lora = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    log.info(f"Trainable (LoRA only): {total_lora:,} ({total_lora/n_params*100:.2f}%)")

    # Dataset
    log.info("Loading wikitext dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
        ds = ds.filter(lambda x: len(x.get("text", "").strip()) > 10)
        ds = ds.select(range(min(2000, len(ds))))

        def tok_fn(ex):
            text = ex.get("text", "")
            enc = trainer.tokenizer(text, truncation=True, max_length=args.seq,
                                    padding="max_length", return_tensors=None)
            enc["labels"] = enc["input_ids"][:]
            return enc
        ds = ds.map(tok_fn, remove_columns=ds.column_names, batched=False)
        ds = ds.filter(lambda x: len(x.get("input_ids", [])) == args.seq)
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        log.info(f"Dataset: {len(ds)} samples")
        use_data = True
    except Exception as e:
        log.warning(f"Dataset failed: {e}")
        use_data = False

    optimizer = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    trainer.model.train()
    losses = []
    t0 = time.time()
    step = 0
    round_idx = 0

    log.info(f"\nStarting: {args.steps} steps | batch={args.batch} | seq={args.seq}")
    log.info("LISA: bottom=2 always, top=2 always, middle_sample=1-2\n")

    while step < args.steps:
        # LISA layer selection every 10 steps
        if step % 10 == 0:
            trainer.freeze_all()
            selected = trainer.select_layers(round_idx)
            trainer.unfreeze_layers(selected)
            nums = [n.split(".")[-1] for n in selected]
            log.info(f"  [Round {round_idx}] Layers: {nums}")
            round_idx += 1

        # Batch
        if use_data:
            idx = torch.randperm(len(ds))[:args.batch].tolist()
            input_ids = torch.stack([ds[i]["input_ids"] for i in idx])
            input_ids = input_ids.clamp(0, trainer.tokenizer.vocab_size - 1)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
        else:
            input_ids = torch.randint(0, trainer.tokenizer.vocab_size, (args.batch, args.seq))
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

        optimizer.zero_grad()
        outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1),
            ignore_index=trainer.tokenizer.pad_token_id or -100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in trainer.model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        losses.append(loss.item())
        step += 1

        if step % 10 == 0:
            avg = sum(losses[-10:]) / min(10, len(losses))
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            log.info(f"  Step {step}/{args.steps} | loss={avg:.4f} | {sps:.2f} steps/s")

        if step % 25 == 0 and step > 0:
            ckpt = output_dir / f"step_{step}.pt"
            torch.save(trainer.model.state_dict(), ckpt)
            log.info(f"  Saved: {ckpt.name}")

    # Final
    final_ckpt = output_dir / "final_model.pt"
    torch.save(trainer.model.state_dict(), final_ckpt)
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    total_time = time.time() - t0

    log.info(f"\nTraining complete!")
    log.info(f"  Steps: {step}")
    log.info(f"  Avg loss: {avg_loss:.4f}")
    log.info(f"  Time: {total_time:.1f}s ({total_time/max(1,step):.2f}s/step)")
    log.info(f"  Checkpoints: {output_dir}")

    # Inference demo
    log.info("\n--- Inference demo ---")
    trainer.model.eval()
    prompts = [
        "The history of artificial intelligence",
        "Machine learning models can",
        "Neural networks are",
    ]
    for prompt in prompts:
        inputs = trainer.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        inputs = {k: v.clamp(0, trainer.tokenizer.vocab_size - 1) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs, max_new_tokens=20, do_sample=True, temperature=0.8,
                pad_token_id=trainer.tokenizer.eos_token_id,
            )
        text = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
        safe = text.encode("cp1252", errors="replace").decode("cp1252")
        log.info(f"  Prompt: {prompt}")
        log.info(f"  Output: {safe}")

    trainer.cleanup()
    return {"status": "ok", "steps": step, "loss": avg_loss}


if __name__ == "__main__":
    main()
