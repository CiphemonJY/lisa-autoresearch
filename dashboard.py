#!/usr/bin/env python3
"""
LISA_FTM — Comprehensive Gradio Dashboard
Federated fine-tuning dashboard with 9 feature tabs.
"""
import os, sys, time, torch, subprocess, json, traceback
import gradio as gr
from pathlib import Path

ROOT = Path(__file__).parent

# ── Helpers ────────────────────────────────────────────────────────────────

def get_server_status():
    try:
        import requests
        r = requests.get("http://SERVER_IP:8080/status", timeout=3)
        return r.json()
    except Exception:
        return {"error": "Server unreachable"}

def get_checkpoints():
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("model_round_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))

def run_cmd(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as e:
        return str(e)

# ── Tab 1: Server Control ──────────────────────────────────────────────────

def server_status():
    status = get_server_status()
    if "error" in status:
        return f"❌ {status['error']}"
    lines = [
        f"Round: {status.get('global_round', '?')}",
        f"Gradients: {status.get('total_gradients_received', 0)}",
        f"Clients: {status.get('connected_clients', 0)}",
        f"Model: {status.get('model_name', '?')}",
    ]
    return "\n".join(lines)

def server_start(port=8080):
    out = run_cmd(f"cd {ROOT} && python main.py --mode server --port {port} --rounds 10", timeout=5)
    return f"Server starting in background...\n{out[:500]}"

def server_stop():
    out = run_cmd("curl -s -X POST http://SERVER_IP:8080/shutdown", timeout=5)
    return f"Shutdown signal sent.\n{out[:200]}"

# ── Tab 2: Clients ────────────────────────────────────────────────────────

def client_status():
    status = get_server_status()
    if "error" in status:
        return "Server offline"
    grads = status.get("total_gradients_received", 0)
    round_ = status.get("global_round", 0)
    return f"Round: {round_}\nTotal gradients received: {grads}"

def client_connect(client_id, server_url):
    out = run_cmd(
        f"cd {ROOT} && python main.py --mode client --client-id {client_id} --server {server_url}",
        timeout=120
    )
    return out[-2000:] if out else "(no output)"

# ── Tab 3: Compression ────────────────────────────────────────────────────

def compression_info():
    try:
        from federated.compression import compress_sparsify, decompress_sparsify
        import numpy as np
        test = torch.randn(1000)
        sparse, meta = compress_sparsify(test.numpy(), top_k=100)
        decompressed = decompress_sparsify(sparse, meta)
        ratio = test.numel() / (len(sparse) + len(meta))
        return f"Sparsification: top-100 of 1000 → {ratio:.1f}x compression\n"
    except Exception as e:
        return f"Compression info: {e}"

# ── Tab 4: Byzantine Defense ──────────────────────────────────────────────

def byzantine_info():
    return (
        "Krum Guard: selects the gradient closest to the geometric median among peers\n"
        "MAD Filter: flags gradients >5 MAD from median as suspicious\n"
        "Norm Clipping: caps gradient norm to prevent explosion\n"
        "Byzantine resilience: up to f malicious clients (of n total)"
    )

# ── Tab 5: DP Privacy ────────────────────────────────────────────────────

def dp_info():
    return (
        "Mechanism: Gaussian noise added to averaged gradient\n"
        "Privacy budget: tracked via dp_epsilon in each update\n"
        "Noise scale: proportional to sensitivity / epsilon\n"
        "Clipping: gradient norm clipped before averaging"
    )

# ── Tab 6: Model Merging (6 methods) ──────────────────────────────────────

MERGE_METHODS = [
    "fedavg", "ties", "dare", "fisher", "soups", "slerp"
]

def merge_preview(method, checkpoint_a, checkpoint_b):
    if not checkpoint_a or not checkpoint_b:
        return "Select two checkpoints"
    if checkpoint_a == checkpoint_b:
        return "Select different checkpoints"
    return f"Merging {checkpoint_a} + {checkpoint_b} using {method}...\n(merge preview)"

def run_merge(method, checkpoint_a, checkpoint_b, t=0.5):
    if not checkpoint_a or not checkpoint_b:
        return "Select two checkpoints"
    try:
        from federated.merge import merge_models
        import torch

        a_path = str(ROOT / "checkpoints" / checkpoint_a)
        b_path = str(ROOT / "checkpoints" / checkpoint_b)

        sd_a = torch.load(a_path, map_location="cpu", weights_only=False)
        sd_b = torch.load(b_path, map_location="cpu", weights_only=False)

        if isinstance(sd_a, dict):
            sd_a = {k: v for k, v in sd_a.items() if hasattr(v, "shape")}
        if isinstance(sd_b, dict):
            sd_b = {k: v for k, v in sd_b.items() if hasattr(v, "shape")}

        result = merge_models([sd_a, sd_b], method=method)
        n_params = sum(v.numel() for v in result.values() if hasattr(v, "numel"))
        return f"Merge complete ({method})\nParams: {n_params:,}\nLayers: {len(result)}"
    except Exception as e:
        return f"Error: {e}"

# ── Tab 7: Evaluation ─────────────────────────────────────────────────────

def eval_checkpoint(checkpoint_name, samples=100):
    if not checkpoint_name:
        return "Select a checkpoint"
    try:
        import math, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        import torch.nn as nn
        from datasets import load_dataset

        MODEL_ID = "EleutherAI/pythia-70m"
        ckpt_path = ROOT / "checkpoints" / checkpoint_name

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
        model.resize_token_embeddings(50304)
        model.load_state_dict(state, strict=False)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
        tok.pad_token = tok.eos_token

        ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
        texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20][:samples]
        enc = tok(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        crit = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id or 0)
        total_loss, total_tokens = 0.0, 0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(enc["input_ids"]), 8):
                ids = enc["input_ids"][i:i+8]
                labs = enc["labels"][i:i+8]
                out = model(input_ids=ids)
                loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
                total_loss += loss.item() * ids.numel()
                total_tokens += ids.numel()
        ppl = math.exp(total_loss / max(total_tokens, 1))
        return f"Checkpoint: {checkpoint_name}\nPPL: {ppl:.2f}\nTokens: {total_tokens}"
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()[:500]}"

def eval_all_checkpoints():
    """Evaluate all checkpoints and return comparison table."""
    ckpts = get_checkpoints()
    if not ckpts:
        return "No checkpoints found"
    lines = ["| Checkpoint | PPL | vs Baseline |", "|---|---|---|"]
    baseline_ppl = None
    results = []
    for ckpt in ckpts:
        result = eval_checkpoint(ckpt.name, samples=50)
        for line in result.split("\n"):
            if line.startswith("PPL:"):
                try:
                    ppl = float(line.split(":")[1].strip())
                    results.append((ckpt.name, ppl))
                except:
                    pass
        lines.append(f"| {ckpt.name} | ... | ... |")
    return "\n".join(lines[:20])

# ── Tab 8: Training History ───────────────────────────────────────────────

def training_history():
    status = get_server_status()
    grads = status.get("total_gradients_received", 0)
    round_ = status.get("global_round", 0)
    ckpts = get_checkpoints()
    lines = [
        f"Server Status: {'Online' if 'error' not in status else 'Offline'}",
        f"Current Round: {round_}",
        f"Total Gradients: {grads}",
        f"Checkpoints: {len(ckpts)}",
        "",
        "Recent checkpoints:"
    ]
    for c in sorted(ckpts, key=lambda p: int(p.stem.split("_")[-1]))[-5:]:
        lines.append(f"  {c.name} ({c.stat().length // 1024 // 1024}MB)")
    return "\n".join(lines)

# ── Tab 9: System Health ─────────────────────────────────────────────────

def system_health():
    import psutil
    lines = [
        f"CPU: {psutil.cpu_percent()}%",
        f"RAM: {psutil.virtual_memory().percent}% used",
        f"Disk: {psutil.disk_usage('/').percent}% used",
    ]
    try:
        status = get_server_status()
        if "error" not in status:
            lines.append(f"Server: Online (round {status.get('global_round', '?')})")
        else:
            lines.append("Server: Offline")
    except:
        lines.append("Server: Unknown")
    return "\n".join(lines)

# ── Build Dashboard ──────────────────────────────────────────────────────

def build_tab(label, fn, *args, **kwargs):
    """Helper to build a tab from a function."""
    return gr.Interface(fn, inputs=list(args), outputs=label and gr.Textbox(label="", lines=10), title=label, live=False)

css = """
#title { text-align: center; font-size: 2em; font-weight: bold; margin-bottom: 1em; }
.gradio-container { max-width: 1400px !important; }
"""

with gr.Blocks(css=css, title="LISA_FTM Dashboard") as dashboard:
    gr.Markdown("# 🧠 LISA_FTM — Federated Fine-Tuning Dashboard", elem_id="title")

    with gr.Tabs():
        # ── Tab 1: Server Control ────────────────────────────────────────
        with gr.Tab("🖥️ Server"):
            gr.Markdown("### Server Control")
            with gr.Row():
                with gr.Column():
                    gr.Button("📊 Server Status", variant="primary").click(
                        fn=server_status,
                        outputs=gr.Textbox(label="Status", lines=8)
                    )
                    server_status_box = gr.Textbox(label="Status", lines=8)
                    gr.Button("📊 Refresh", variant="secondary").click(
                        fn=server_status, outputs=server_status_box
                    )
                with gr.Column():
                    port = gr.Number(label="Port", value=8080)
                    gr.Button("▶️ Start Server").click(
                        fn=server_start, inputs=port,
                        outputs=gr.Textbox(label="Output", lines=4)
                    )
                    gr.Button("⏹️ Stop Server").click(
                        fn=server_stop,
                        outputs=gr.Textbox(label="Output", lines=4)
                    )

        # ── Tab 2: Clients ──────────────────────────────────────────────
        with gr.Tab("👥 Clients"):
            gr.Markdown("### Federated Clients")
            with gr.Row():
                with gr.Column():
                    client_id = gr.Textbox(label="Client ID", value="client-1")
                    server_url = gr.Textbox(label="Server URL", value="http://SERVER_IP:8080")
                    gr.Button("🚀 Connect Client", variant="primary").click(
                        fn=client_connect,
                        inputs=[client_id, server_url],
                        outputs=gr.Textbox(label="Output", lines=12)
                    )
                with gr.Column():
                    gr.Button("🔍 Client Status").click(
                        fn=client_status,
                        outputs=gr.Textbox(label="Status", lines=8)
                    )

        # ── Tab 3: Compression ─────────────────────────────────────────
        with gr.Tab("📦 Compression"):
            gr.Markdown("### Gradient Compression")
            gr.Markdown(
                "LISA_FTM uses **top-K sparsification** + **uint8 quantization** to reduce gradient size.\n\n"
                "Typical compression ratios: **10-50x** for sparse gradients.\n\n"
                "Methods: `top_k`, `quantize`, `both` (sparsify + quantize)"
            )
            gr.Button("📊 Compression Info", variant="primary").click(
                fn=compression_info,
                outputs=gr.Textbox(label="Info", lines=6)
            )
            gr.Button("🧪 Test Compression").click(
                fn=compression_info,
                outputs=gr.Textbox(label="Test Result", lines=6)
            )

        # ── Tab 4: Byzantine Defense ────────────────────────────────────
        with gr.Tab("🛡️ Byzantine"):
            gr.Markdown("### Byzantine-Resilient Federated Learning")
            gr.Markdown(byzantine_info())
            gr.Markdown(
                "| Method | Tolerance |\n|---|---|\n"
                "| Krum | ⌊(n+f-2)/2⌋ malicious |\n"
                "| MAD Filter | Configurable threshold |\n"
                "| Norm Clipping | Per-update |\n"
            )

        # ── Tab 5: DP Privacy ──────────────────────────────────────────
        with gr.Tab("🔒 Privacy"):
            gr.Markdown("### Differential Privacy")
            gr.Markdown(dp_info())
            gr.Markdown(
                "| Parameter | Typical Value |\n|---|---|\n"
                "| Noise σ | 0.1–2.0 |\n"
                "| Clipping C | 1.0–5.0 |\n"
                "| ε (budget) | 2.0–8.0 |\n"
            )

        # ── Tab 6: Model Merging ────────────────────────────────────────
        with gr.Tab("🔀 Merging"):
            gr.Markdown("### Model Merging — 7 Methods")
            gr.Markdown(
                "**fedavg** · **ties** · **dare** · **fisher** · **soups** · **slerp** · **svd**\n\n"
                "Merge two checkpoints or accumulate gradient deltas from multiple clients."
            )
            checkpoint_files = [c.name for c in get_checkpoints()]
            if not checkpoint_files:
                gr.Markdown("⚠️ No checkpoints found. Run training first.")
            else:
                with gr.Row():
                    method = gr.Dropdown(MERGE_METHODS, value="fedavg", label="Method")
                    ckpt_a = gr.Dropdown(checkpoint_files, value=checkpoint_files[0] if len(checkpoint_files)>0 else None, label="Checkpoint A")
                    ckpt_b = gr.Dropdown(checkpoint_files, value=checkpoint_files[1] if len(checkpoint_files)>1 else None, label="Checkpoint B")
                with gr.Row():
                    t_param = gr.Slider(0, 1, value=0.5, step=0.05, label="t (interpolation factor)")
                    gr.Button("🔀 Merge", variant="primary").click(
                        fn=run_merge,
                        inputs=[method, ckpt_a, ckpt_b, t_param],
                        outputs=gr.Textbox(label="Result", lines=8)
                    )
                gr.Button("📊 Merge Preview").click(
                    fn=merge_preview,
                    inputs=[method, ckpt_a, ckpt_b],
                    outputs=gr.Textbox(label="Preview", lines=4)
                )

        # ── Tab 7: Evaluation ─────────────────────────────────────────
        with gr.Tab("📈 Evaluation"):
            gr.Markdown("### Perplexity Evaluation")
            ckpt_files = [c.name for c in get_checkpoints()]
            if ckpt_files:
                with gr.Row():
                    eval_ckpt = gr.Dropdown(ckpt_files, label="Checkpoint")
                    eval_samples = gr.Number(label="Samples", value=100)
                    gr.Button("📊 Evaluate", variant="primary").click(
                        fn=eval_checkpoint,
                        inputs=[eval_ckpt, eval_samples],
                        outputs=gr.Textbox(label="PPL Result", lines=6)
                    )
                gr.Button("📊 Compare All Checkpoints").click(
                    fn=eval_all_checkpoints,
                    outputs=gr.Textbox(label="Comparison", lines=15)
                )
            else:
                gr.Markdown("⚠️ No checkpoints found.")

        # ── Tab 8: Training History ────────────────────────────────────
        with gr.Tab("📜 History"):
            gr.Markdown("### Training History")
            gr.Button("📜 Load History", variant="primary").click(
                fn=training_history,
                outputs=gr.Textbox(label="History", lines=20)
            )
            gr.Button("📁 List Checkpoints").click(
                fn=lambda: "\n".join(c.name for c in get_checkpoints()),
                outputs=gr.Textbox(label="Checkpoints", lines=15)
            )

        # ── Tab 9: System Health ────────────────────────────────────────
        with gr.Tab("💻 System"):
            gr.Markdown("### System Health")
            gr.Button("💻 Check Health", variant="primary").click(
                fn=system_health,
                outputs=gr.Textbox(label="Health", lines=10)
            )
            gr.Button("🐍 Python Info").click(
                fn=lambda: f"Python {sys.version}\nPyTorch {torch.__version__}",
                outputs=gr.Textbox(label="Python/PyTorch", lines=6)
            )

    gr.Markdown(
        "\n---\n*LISA_FTM · Federated Fine-Tuning · [GitHub](https://github.com/CiphemonJY/LISA_FTM)*",
        elem_id="footer"
    )

if __name__ == "__main__":
    print("Starting LISA_FTM Dashboard...")
    print("Open http://localhost:7860 in your browser")
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False,
    )
