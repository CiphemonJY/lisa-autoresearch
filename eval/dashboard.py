#!/usr/bin/env python3
"""
LISA-FTM Gradio Dashboard

Usage:
  pip install gradio pandas numpy
  python eval/dashboard.py     # Opens http://localhost:7860
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_GLOB = str(RESULTS_DIR / "exp_*.json")

# ---------------------------------------------------------------------------
# Results loading helpers
# ---------------------------------------------------------------------------

def load_all_experiments() -> List[Dict[str, Any]]:
    """Load all exp_*.json files from eval/results/."""
    experiments = []
    for path in sorted(RESULTS_DIR.glob("exp_*.json")):
        try:
            with open(path, "r") as f:
                experiments.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    return experiments


def load_layer_stats() -> Optional[Dict[str, Any]]:
    """Load layer selection stats from the comparative eval or from experiments."""
    # First try the dedicated layer stats file
    layer_path = RESULTS_DIR / "layer_selection_stats.json"
    if layer_path.exists():
        try:
            with open(layer_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Fall back: aggregate from all experiments
    experiments = load_all_experiments()
    if not experiments:
        return None

    all_counts: Dict[str, int] = {}
    for exp in experiments:
        for k, v in exp.get("layer_counts", {}).items():
            all_counts[k] = all_counts.get(k, 0) + v

    if not all_counts:
        return None

    return {
        "layer_counts": all_counts,
        "source": "aggregated",
    }


def load_comparison_results() -> Optional[Dict[str, Any]]:
    """Load fedavg_vs_lisafedavg.json if it exists."""
    path = RESULTS_DIR / "fedavg_vs_lisafedavg.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


# ---------------------------------------------------------------------------
# Data shaping helpers
# ---------------------------------------------------------------------------

def experiments_table_data(
    experiments: List[Dict[str, Any]],
    compression_filter: Optional[str] = None,
    middle_sample_filter: Optional[int] = None,
) -> List[List[Any]]:
    """
    Return [[exp_id, config_hash, final_ppl, comm_cost, ...], ...]
    """
    rows = []
    for exp in experiments:
        cfg = exp.get("config", {})
        compression = cfg.get("COMPRESSION", "none")
        middle = cfg.get("LISA_MIDDLE_SAMPLE", None)

        if compression_filter and compression_filter != "all" and compression != compression_filter:
            continue
        if middle_sample_filter is not None and middle != middle_sample_filter:
            continue

        rows.append([
            exp.get("exp_id", "?"),
            exp.get("config_hash", "??"),
            f"{exp.get('final_perplexity', 0):.4f}",
            exp.get("total_comm_cost", 0),
            compression,
            middle,
            cfg.get("LISA_BOTTOM_LAYERS", "?"),
            cfg.get("LISA_TOP_LAYERS", "?"),
            cfg.get("LR", "?"),
            cfg.get("LOCAL_STEPS", "?"),
            cfg.get("NUM_CLIENTS", "?"),
            cfg.get("ROUNDS", "?"),
        ])
    return rows


def get_best_experiment(experiments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the experiment with the lowest final perplexity."""
    if not experiments:
        return None
    return min(experiments, key=lambda e: e.get("final_perplexity", float("inf")))


def get_unique_values(experiments: List[Dict[str, Any]], key: str) -> List[Any]:
    """Extract unique values for a config key across all experiments."""
    vals = set()
    for exp in experiments:
        v = exp.get("config", {}).get(key)
        if v is not None:
            vals.add(v)
    return sorted(vals)


# ---------------------------------------------------------------------------
# Live server status (placeholder — checks for a running server process)
# ---------------------------------------------------------------------------

def read_server_status() -> Dict[str, Any]:
    """
    Attempt to read live server status.
    Looks for eval/server_status.json written by a running server process.
    Returns empty dict if no server is running.
    """
    status_path = EVAL_DIR / "server_status.json"
    if status_path.exists():
        try:
            with open(status_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Background process runner for Quick Run tab
# ---------------------------------------------------------------------------

class BackgroundRunner:
    """Manages a background subprocess for the Quick Run tab."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.output_lines: List[str] = []
        self._lock = threading.Lock()
        self._done = threading.Event()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, cfg_updates: Optional[Dict[str, Any]] = None):
        if self.is_running():
            return

        self._done.clear()
        self.output_lines = []

        def write_cfg():
            if cfg_updates:
                cfg_path = EVAL_DIR / "fed_config.py"
                lines = []
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        lines = f.readlines()
                # Simple approach: append/override config values
                # Write a minimal fed_config if it doesn't exist
                if not cfg_path.exists():
                    lines = ["# Auto-generated by dashboard\n"]
                # Update config values
                import ast, inspect
                try:
                    current = {}
                    if cfg_path.exists():
                        with open(cfg_path) as f:
                            src = f.read()
                    else:
                        src = ""
                    # Parse existing
                    tree = ast.parse(src) if src else ast.Module()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            for t in node.targets:
                                if isinstance(t, ast.Name):
                                    try:
                                        current[t.id] = ast.literal_eval(node.value)
                                    except Exception:
                                        pass
                except Exception:
                    current = {}

                current.update(cfg_updates or {})
                with open(cfg_path, "w") as f:
                    f.write("# Auto-generated by dashboard\n")
                    for k, v in current.items():
                        f.write(f"{k} = {repr(v)}\n")
            else:
                # Just ensure fed_config.py exists with defaults
                cfg_path = EVAL_DIR / "fed_config.py"
                if not cfg_path.exists():
                    with open(cfg_path, "w") as f:
                        f.write("# Default config\nMODEL = 'EleutherAI/pythia-70m'\n"
                                "NUM_CLIENTS = 3\nROUNDS = 5\nLOCAL_STEPS = 5\n"
                                "LR = 0.0003\nLISA_BOTTOM_LAYERS = 2\n"
                                "LISA_TOP_LAYERS = 2\nLISA_MIDDLE_SAMPLE = 2\n"
                                "COMPRESSION = 'none'\nCOMPRESSION_K = 0.1\n"
                                "COMPRESSION_BITS = 8\nNOISE_MULTIPLIER = 0.0\n")

        write_cfg()

        env = os.environ.copy()
        self.process = subprocess.Popen(
            [sys.executable, str(EVAL_DIR / "autora.py"), "run"],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        def stream():
            try:
                for line in self.process.stdout:
                    if line is None:
                        break
                    with self._lock:
                        self.output_lines.append(line.rstrip())
            except Exception:
                pass
            finally:
                self._done.set()

        t = threading.Thread(target=stream, daemon=True)
        t.start()

    def get_output(self) -> str:
        with self._lock:
            return "\n".join(self.output_lines)

    def stop(self):
        if self.process and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def poll_done(self) -> bool:
        return self._done.is_set()


runner = BackgroundRunner()


# ---------------------------------------------------------------------------
# Gradio UI — Build
# ---------------------------------------------------------------------------

def build_dashboard():
    css = """
    .gradio-container { max-width: 1400px !important; margin: auto !important; }
    .tab-title { font-size: 1.2em; font-weight: 600; }
    .status-ok { color: #22c55e; }
    .status-warn { color: #f59e0b; }
    .status-bad { color: #ef4444; }
    .best-row { background-color: #fef9c3 !important; }
    pre.log-output { background: #1e1e1e; color: #d4d4d4; padding: 12px;
                     border-radius: 8px; font-size: 12px; max-height: 500px;
                     overflow-y: auto; font-family: 'Courier New', monospace; }
    """

    with gr.Blocks(css=css, title="LISA-FTM Dashboard") as demo:
        gr.Markdown("# 🔬 LISA-FTM Autoresearch Dashboard")
        gr.Markdown(
            "Federated fine-tuning experiments with LISA layer selection. "
            "Explore results, visualize layer selection, and run new experiments."
        )

        tabs = gr.Tabs()

        # ──────────────────────────────────────────────────────────────
        # Tab 1: Experiment Results
        # ──────────────────────────────────────────────────────────────
        with tabs.tab("📊 Experiment Results"):
            gr.Markdown("### Experiment Results")

            with gr.Row():
                compression_filter = gr.Dropdown(
                    label="Compression Filter",
                    choices=["all", "none", "sparsify", "quantize", "both"],
                    value="all",
                    interactive=True,
                )
                middle_sample_filter = gr.Dropdown(
                    label="LISA Middle Sample Filter",
                    choices=[None, 0, 1, 2, 3, 4, 5],
                    value=None,
                    interactive=True,
                    allow_none_value=True,
                )
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                refresh_btn.style(size="sm")

            with gr.Row():
                ppl_plot = gr.LinePlot(
                    label="Perplexity per Round",
                    x="Round",
                    y="Perplexity",
                    tooltip=["Round", "Perplexity", "Experiment"],
                    height=350,
                )
                comm_plot = gr.LinePlot(
                    label="Communication Cost per Round",
                    x="Round",
                    y="Comm Cost",
                    tooltip=["Round", "Comm Cost", "Experiment"],
                    height=350,
                )

            with gr.Row():
                best_exp_display = gr.JSON(
                    label="🏆 Best Experiment (lowest perplexity)",
                    visible=True,
                )

            table_state = gr.State(value=[])

            results_table = gr.DataFrame(
                headers=[
                    "Exp ID", "Hash", "Final PPL", "Comm Cost",
                    "Compression", "LISA Middle", "LISA Bottom", "LISA Top",
                    "LR", "Local Steps", "Clients", "Rounds",
                ],
                label="All Experiments",
                interactive=False,
                visible=True,
                wrap=True,
            )

            def update_results(
                compression: str,
                middle_sample: Optional[int],
            ):
                experiments = load_all_experiments()
                table_state_val = experiments

                if not experiments:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=None),
                        gr.update(value=[], visible=True),
                        experiments,
                    )

                # Filter
                filtered = experiments
                if compression != "all":
                    filtered = [
                        e for e in filtered
                        if e.get("config", {}).get("COMPRESSION", "none") == compression
                    ]
                if middle_sample is not None:
                    filtered = [
                        e for e in filtered
                        if e.get("config", {}).get("LISA_MIDDLE_SAMPLE") == middle_sample
                    ]

                table = experiments_table_data(experiments)
                best = get_best_experiment(experiments)
                best_val = {
                    "exp_id": best.get("exp_id") if best else None,
                    "final_perplexity": f"{best.get('final_perplexity', 0):.4f}" if best else None,
                    "total_comm_cost": best.get("total_comm_cost") if best else None,
                    "config": best.get("config") if best else None,
                    "perplexity_curve": best.get("perplexity_per_round") if best else None,
                    "config_hash": best.get("config_hash") if best else None,
                }

                # Build LinePlot data
                # Each experiment = one line
                # x = rounds, y = perplexity, color = exp_id
                import pandas as pd

                ppl_rows = []
                comm_rows = []
                for exp in filtered:
                    exp_id = exp.get("exp_id", "?")
                    rounds = exp.get("perplexity_per_round", [])
                    comms = exp.get("comm_cost_per_round", [])
                    for i, (ppl, comm) in enumerate(zip(rounds, comms), 1):
                        ppl_rows.append({"Round": i, "Perplexity": ppl, "Experiment": exp_id})
                        comm_rows.append({"Round": i, "Comm Cost": comm, "Experiment": exp_id})

                ppl_df = pd.DataFrame(ppl_rows) if ppl_rows else pd.DataFrame(columns=["Round", "Perplexity", "Experiment"])
                comm_df = pd.DataFrame(comm_rows) if comm_rows else pd.DataFrame(columns=["Round", "Comm Cost", "Experiment"])

                return (
                    gr.update(value=ppl_df, visible=True),
                    gr.update(value=comm_df, visible=True),
                    gr.update(value=best_val, visible=True),
                    gr.update(value=table, visible=True),
                    filtered,
                )

            def refresh_callback(compression, middle_sample):
                return update_results(compression, middle_sample)

            refresh_btn.click(
                fn=refresh_callback,
                inputs=[compression_filter, middle_sample_filter],
                outputs=[ppl_plot, comm_plot, best_exp_display, results_table, table_state],
            )

            # Load on tab show
            compression_filter.change(
                fn=refresh_callback,
                inputs=[compression_filter, middle_sample_filter],
                outputs=[ppl_plot, comm_plot, best_exp_display, results_table, table_state],
            )
            middle_sample_filter.change(
                fn=refresh_callback,
                inputs=[compression_filter, middle_sample_filter],
                outputs=[ppl_plot, comm_plot, best_exp_display, results_table, table_state],
            )

            # Initial load
            demo.load(
                fn=lambda: update_results("all", None),
                inputs=[],
                outputs=[ppl_plot, comm_plot, best_exp_display, results_table, table_state],
            )

        # ──────────────────────────────────────────────────────────────
        # Tab 2: Layer Selection Heatmap
        # ──────────────────────────────────────────────────────────────
        with tabs.tab("🗺️ Layer Selection Heatmap"):
            gr.Markdown("### Layer Selection Frequency Across All Experiments")

            heatmap_state = gr.State(value=None)

            with gr.Row():
                heatmap_plot = gr.Heatmap(
                    label="Layer Selection Heatmap",
                    height=400,
                )
                layer_stats_json = gr.JSON(label="Layer Statistics")

            heatmap_refresh = gr.Button("🔄 Refresh Heatmap", variant="secondary")

            def update_heatmap():
                layer_stats = load_layer_stats()
                comparison = load_comparison_results()

                # Try to build a heatmap from layer selection counts
                # Rows = layer index, Cols = experiment/round
                # Color = selection count
                import pandas as pd

                experiments = load_all_experiments()
                if not experiments:
                    if comparison and "layer_counts" in comparison:
                        counts = comparison.get("layer_counts", {})
                        max_layer = max(int(k) for k in counts.keys()) + 1 if counts else 0
                        rows = []
                        for layer_idx in range(max_layer):
                            for exp_label in ["LISA-FedAvg"]:
                                rows.append({
                                    "Layer": layer_idx,
                                    "Experiment": exp_label,
                                    "Selections": counts.get(str(layer_idx), 0),
                                })
                        df = pd.DataFrame(rows)
                        stats = {
                            "layer_counts": counts,
                            "total_rounds": comparison.get("num_rounds", "?"),
                            "total_clients": comparison.get("num_clients", "?"),
                            "lisa_config": comparison.get("lisa_config", {}),
                        }
                        return gr.update(value=df, visible=True), gr.update(value=stats, visible=True)
                    return gr.update(visible=False), gr.update(value=None, visible=True)

                # Build heatmap from experiments
                all_layer_counts: Dict[int, Dict[str, int]] = {}
                for exp in experiments:
                    exp_id = exp.get("exp_id", "?")
                    for k, v in exp.get("layer_counts", {}).items():
                        layer_idx = int(k)
                        if layer_idx not in all_layer_counts:
                            all_layer_counts[layer_idx] = {}
                        all_layer_counts[layer_idx][exp_id] = v

                if not all_layer_counts:
                    return gr.update(visible=False), gr.update(value=None, visible=True)

                max_layer = max(all_layer_counts.keys())
                rows = []
                for layer_idx in range(max_layer + 1):
                    counts_dict = all_layer_counts.get(layer_idx, {})
                    for exp_id in counts_dict:
                        rows.append({
                            "Layer": layer_idx,
                            "Experiment": exp_id,
                            "Selections": counts_dict[exp_id],
                        })

                df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Layer", "Experiment", "Selections"])
                stats = {
                    "layer_counts": {str(k): v for k, v in all_layer_counts.items()},
                    "num_experiments": len(experiments),
                }
                return gr.update(value=df, visible=True), gr.update(value=stats, visible=True)

            heatmap_refresh.click(
                fn=update_heatmap,
                inputs=[],
                outputs=[heatmap_plot, layer_stats_json],
            )
            demo.load(
                fn=update_heatmap,
                inputs=[],
                outputs=[heatmap_plot, layer_stats_json],
            )

        # ──────────────────────────────────────────────────────────────
        # Tab 3: Live Server Status
        # ──────────────────────────────────────────────────────────────
        with tabs.tab("🖥️ Live Server Status"):
            gr.Markdown("### Server Status")

            with gr.Row():
                status_indicator = gr.Markdown("⚪ Unknown")
                server_refresh = gr.Button("🔄 Refresh Status", variant="secondary")

            status_details = gr.JSON(label="Server Details")

            server_running = gr.Checkbox(label="Server Running", interactive=False, value=False)

            with gr.Row():
                connected_clients = gr.Number(label="Connected Clients", value=0, interactive=False)
                current_round = gr.Number(label="Current Round", value=0, interactive=False)
                latest_ppl = gr.Number(label="Latest Perplexity", value=0, interactive=False)
                server_uptime = gr.Textbox(label="Uptime", value="N/A", interactive=False)

            def update_server_status():
                status = read_server_status()
                if not status:
                    # No server status file found — show placeholder
                    return (
                        gr.update(value="⚪ **No server detected**\nRun an experiment to start a server.", visible=True),
                        gr.update(value=None, visible=True),
                        gr.update(value=False, visible=True),
                        gr.update(value=0, visible=True),
                        gr.update(value=0, visible=True),
                        gr.update(value=0.0, visible=True),
                        gr.update(value="N/A", visible=True),
                    )

                running = status.get("running", False)
                indicator = "🟢 **Server Running**" if running else "⚪ **Server Stopped**"
                return (
                    gr.update(value=indicator, visible=True),
                    gr.update(value=status, visible=True),
                    gr.update(value=running, visible=True),
                    gr.update(value=status.get("connected_clients", 0), visible=True),
                    gr.update(value=status.get("current_round", 0), visible=True),
                    gr.update(value=status.get("latest_perplexity", 0.0), visible=True),
                    gr.update(value=status.get("uptime", "N/A"), visible=True),
                )

            server_refresh.click(
                fn=update_server_status,
                inputs=[],
                outputs=[
                    status_indicator,
                    status_details,
                    server_running,
                    connected_clients,
                    current_round,
                    latest_ppl,
                    server_uptime,
                ],
            )
            demo.load(
                fn=update_server_status,
                inputs=[],
                outputs=[
                    status_indicator,
                    status_details,
                    server_running,
                    connected_clients,
                    current_round,
                    latest_ppl,
                    server_uptime,
                ],
            )

        # ──────────────────────────────────────────────────────────────
        # Tab 4: Quick Run
        # ──────────────────────────────────────────────────────────────
        with tabs.tab("⚡ Quick Run"):
            gr.Markdown("### Run a New Experiment")
            gr.Markdown(
                "Configure and launch an experiment. It runs in the background — "
                "output streams live below."
            )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    value="EleutherAI/pythia-70m",
                    interactive=True,
                    allow_edit=True,
                )
                num_clients = gr.Number(
                    label="Number of Clients",
                    value=3,
                    interactive=True,
                    precision=0,
                )
                num_rounds = gr.Number(
                    label="Number of Rounds",
                    value=5,
                    interactive=True,
                    precision=0,
                )

            with gr.Row():
                lisa_bottom = gr.Number(
                    label="LISA Bottom Layers",
                    value=2,
                    interactive=True,
                    precision=0,
                )
                lisa_top = gr.Number(
                    label="LISA Top Layers",
                    value=2,
                    interactive=True,
                    precision=0,
                )
                lisa_middle = gr.Number(
                    label="LISA Middle Sample",
                    value=2,
                    interactive=True,
                    precision=0,
                )

            with gr.Row():
                lr_slider = gr.Number(
                    label="Learning Rate",
                    value=0.0003,
                    interactive=True,
                )
                local_steps = gr.Number(
                    label="Local Steps",
                    value=5,
                    interactive=True,
                    precision=0,
                )
                compression_dropdown = gr.Dropdown(
                    label="Compression",
                    value="none",
                    choices=["none", "sparsify", "quantize", "both"],
                    interactive=True,
                )

            with gr.Row():
                run_btn = gr.Button("🚀 Run Experiment", variant="primary")
                stop_btn = gr.Button("⏹ Stop", variant="stop")
                clear_btn = gr.Button("🗑 Clear Output", variant="secondary")

            output_text = gr.Textbox(
                label="Live Output",
                lines=20,
                interactive=False,
                show_copy_button=True,
            )
            status_text = gr.Markdown("**Status:** Idle")

            def run_experiment_handler(
                model, num_clients, num_rounds,
                lisa_bottom, lisa_top, lisa_middle,
                lr, local_steps, compression,
            ):
                cfg = {
                    "MODEL": model,
                    "NUM_CLIENTS": int(num_clients),
                    "ROUNDS": int(num_rounds),
                    "LISA_BOTTOM_LAYERS": int(lisa_bottom),
                    "LISA_TOP_LAYERS": int(lisa_top),
                    "LISA_MIDDLE_SAMPLE": int(lisa_middle),
                    "LR": float(lr),
                    "LOCAL_STEPS": int(local_steps),
                    "COMPRESSION": compression,
                    "COMPRESSION_K": 0.1,
                    "COMPRESSION_BITS": 8,
                    "NOISE_MULTIPLIER": 0.0,
                }

                # Write config
                cfg_path = EVAL_DIR / "fed_config.py"
                with open(cfg_path, "w") as f:
                    f.write("# Auto-generated by dashboard\n")
                    for k, v in cfg.items():
                        f.write(f"{k} = {repr(v)}\n")

                # Start background process
                env = os.environ.copy()
                proc = subprocess.Popen(
                    [sys.executable, str(EVAL_DIR / "autora.py"), "run"],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    bufsize=1,
                )

                output_parts = []
                done = False

                def stream_output():
                    try:
                        for line in iter(proc.stdout.readline, ""):
                            if not line:
                                break
                            output_parts.append(line.rstrip())
                            yield "\n".join(output_parts[-500:])  # keep last 500 lines
                    except Exception:
                        pass
                    finally:
                        done = True

                # Stream output by yielding
                # First yield: start message
                yield "🚀 Experiment started! Output will stream below...\n", gr.update(value="🟡 **Running...**", visible=True)

                try:
                    for out_chunk in stream_output():
                        yield out_chunk, gr.update(value="🟡 **Running...**", visible=True)
                finally:
                    proc.wait()
                    retcode = proc.returncode
                    final_msg = (
                        f"\n{'='*60}\n"
                        f"Experiment finished (exit code: {retcode}).\n"
                        f"Results saved to eval/results/. Refresh the Experiment Results tab."
                    )
                    output_parts.append(final_msg)
                    yield "\n".join(output_parts[-500:]), gr.update(
                        value="🟢 **Done!**" if retcode == 0 else f"❌ **Failed (exit {retcode})**",
                        visible=True,
                    )

            def stop_handler():
                runner.stop()
                return "⏹ Stopped.", gr.update(value="❌ **Stopped by user**", visible=True)

            def clear_handler():
                return "", gr.update(value="**Status:** Idle", visible=True)

            run_btn.click(
                fn=run_experiment_handler,
                inputs=[
                    model_dropdown, num_clients, num_rounds,
                    lisa_bottom, lisa_top, lisa_middle,
                    lr_slider, local_steps, compression_dropdown,
                ],
                outputs=[output_text, status_text],
            )
            stop_btn.click(
                fn=stop_handler,
                inputs=[],
                outputs=[output_text, status_text],
            )
            clear_btn.click(
                fn=clear_handler,
                inputs=[],
                outputs=[output_text, status_text],
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Starting LISA-FTM Dashboard...")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Open http://localhost:7860 in your browser")
    print("Press Ctrl+C to stop")

    demo = build_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
