#!/usr/bin/env python3
"""
Real training run on Windows/Intel PC.
Uses pythia-70m (14M params, proper nn.Linear) with LoRA on real eli5 dataset.
"""
import os, sys, time, torch, logging, argparse
from pathlib import Path

# Setup
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("real-train")

import torch
# Detect best device for Apple Silicon Mac
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
print(f"[macOS] Using device: {DEVICE}")

# ─── Model ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="EleutherAI/pythia-160m")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lora-rank", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()

MODEL_ID = args.model   # Now respects --model flag!

# ─── Dataset ────────────────────────────────────────────────────────────────
# Using wikitext for language modeling (no tokenization headaches)
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-v1"    # Small, fast to download
MAX_SEQ_LEN = 128
TRAIN_SIZE = 2000                    # 2000 samples for real training

# ─── LoRA / LISA ────────────────────────────────────────────────────────────
LORA_RANK = args.lora_rank
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["attn", "fc", "proj"]  # Works for Pythia

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = args.batch_size
GRAD_ACCUM = 4                      # Effective batch = 16
LEARNING_RATE = args.lr
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20
TRAIN_STEPS = args.steps           # Now respects --steps flag!
SAVE_EVERY = 50
OUTPUT_DIR = ROOT / "output" / "real_training"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from torch.utils.data import DataLoader, Dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ──────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    log.info(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Load model ─────────────────────────────────────────────────────────
    log.info(f"Loading model: {MODEL_ID}")
    t0 = time.time()
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, trust_remote_code=True, torch_dtype=DTYPE,
    )
    model.resize_token_embeddings(len(tokenizer))
    n_params = sum(p.numel() for p in model.parameters())

    # ── Apply LoRA BEFORE moving to MPS ───────────────────────────────────
    # LoRA params are created as CPU Parameters; we must move the full
    # model to MPS AFTER LoRA application so all params end up on MPS.
    log.info("Applying LoRA...")
    from lisa.train_torch import LoRALinear
    import torch.nn as nn

    lora_count = 0
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if it's an attention or MLP layer
        if not any(tm in full_name for tm in ["attn", "fc", "proj", "dense"]):
            continue

        # Replace with LoRA
        lora = LoRALinear(module, rank=LORA_RANK, alpha=LORA_ALPHA,
                          dropout=LORA_DROPOUT, target_module_name=full_name)
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            try:
                parent = model.get_submodule(parent_name)
                setattr(parent, attr, lora)
                lora_count += 1
            except (KeyError, AttributeError) as e:
                log.debug(f"Could not apply LoRA to {full_name}: {e}")

    log.info(f"  LoRA applied to {lora_count} layers")

    # ── Move model to MPS device ─────────────────────────────────────────
    model = model.to(DEVICE)
    log.info(f"  Model on device: {next(model.parameters()).device}")

    # Freeze all except LoRA
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable params: {trainable:,} ({trainable/n_params*100:.2f}%)")

    # ── Load real dataset ──────────────────────────────────────────────────
    log.info(f"Loading dataset: {DATASET_NAME}/{DATASET_CONFIG}")
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
        log.info(f"  Dataset loaded: {len(ds)} samples")

        # Tokenize
        def tokenize(example):
            text = example.get("text", example.get("sentence", ""))
            if not text or len(text.strip()) < 10:
                return {"input_ids": [], "attention_mask": []}
            enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN,
                           padding="max_length", return_tensors=None)
            enc["labels"] = enc["input_ids"][:]
            return enc

        ds = ds.filter(lambda x: len(x.get("text", "").strip()) > 10 if isinstance(x, dict) else True)
        ds = ds.select(range(min(TRAIN_SIZE, len(ds))))
        ds = ds.map(tokenize, remove_columns=ds.column_names, batched=False)
        ds = ds.filter(lambda x: len(x.get("input_ids", [])) > 0)
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        log.info(f"  Tokenized: {len(ds)} samples")
    except Exception as e:
        log.warning(f"  Could not load dataset: {e}. Using synthetic data.")
        ds = None

    # ── Optimizer & scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS
    )

    # ── Training loop ───────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    losses = []
    step = 0
    grad_accum_counter = 0

    log.info(f"\nStarting training: {TRAIN_STEPS} steps, effective_batch={BATCH_SIZE*GRAD_ACCUM}")

    while step < TRAIN_STEPS:
        try:
            if ds is not None:
                # Real batch from dataset
                indices = torch.randperm(len(ds))[:BATCH_SIZE].tolist()
                input_ids = torch.stack([ds[i]["input_ids"] for i in indices]).clamp(0, tokenizer.vocab_size - 1)
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
            else:
                # Synthetic batch
                input_ids = torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, MAX_SEQ_LEN))
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()

            outputs = model(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1).to(DEVICE),
                ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
            )
            (loss / GRAD_ACCUM).backward()
            losses.append(loss.item())
            grad_accum_counter += 1

            if grad_accum_counter >= GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_accum_counter = 0
                step += 1

                if step % 10 == 0:
                    avg_loss = sum(losses[-20:]) / min(len(losses), 20)
                    lr = optimizer.param_groups[0]["lr"]
                    log.info(f"  Step {step}/{TRAIN_STEPS} | "
                            f"loss={avg_loss:.4f} | lr={lr:.2e}")

                if step % SAVE_EVERY == 0:
                    ckpt = OUTPUT_DIR / f"step_{step}.pt"
                    torch.save(model.state_dict(), ckpt)
                    log.info(f"  Saved: {ckpt.name}")

        except KeyboardInterrupt:
            log.info("Interrupted by user")
            break

    # ── Final save ─────────────────────────────────────────────────────────
    final = OUTPUT_DIR / "final_model.pt"
    torch.save(model.state_dict(), final)

    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    log.info(f"\nTraining complete!")
    log.info(f"  Final avg loss: {avg_loss:.4f}")
    log.info(f"  Steps trained: {step}")
    log.info(f"  Model saved: {final}")

    # ── Quick inference demo ────────────────────────────────────────────────
    log.info("\n--- Inference demo ---")
    model.eval()
    prompts = [
        "PyTorch is a",
        "Machine learning models",
        "Neural networks can",
    ]
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        inputs = {k: v.clamp(0, tokenizer.vocab_size - 1).to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=15,
                do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        safe = text.encode("cp1252", errors="replace").decode("cp1252")
        log.info(f"  Prompt: {prompt}")
        log.info(f"  Output: {safe}")

    log.info(f"\nOutput dir: {OUTPUT_DIR}")
    return {"status": "ok", "steps": step, "final_loss": avg_loss}


if __name__ == "__main__":
    main()
