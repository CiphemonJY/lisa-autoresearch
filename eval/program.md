# LISA-FTM Autoresearch Program

## What is LISA_FTM?

LISA-FTM (Layer-wise Importance Self-Attention Federated Training Method) combines:

- **Federated Learning**: Multiple clients train a shared model collaboratively without sharing raw data — each client trains locally and only shares gradient updates with a central server.
- **LISA Layer Selection**: Instead of training all layers, LISA trains only the most important layers per round: the bottom K layers, top K layers, and M randomly sampled middle layers. This dramatically reduces communication cost.
- **LoRA (Low-Rank Adaptation)**: Instead of full fine-tuning, LoRA adds small low-rank matrices (A and B) to selected layers. Only the LoRA parameters are trained, reducing both compute and communication.

**Why LISA + Federated?**
- Standard federated learning sends full gradient updates every round → high communication cost
- LISA sends updates for only selected layers → 40–60% less communication with minimal accuracy loss
- LoRA keeps the update size small → additional compression

## The Goal

**Primary metric**: Test perplexity (lower is better)

**Secondary metric**: Communication cost (fewer gradient tensors sent = less bandwidth)

The ideal result: **perplexity as low as possible** while **communication cost as low as possible**.

---

# Experiment Suites

Run each suite in order. Each suite has a clear hypothesis and a set of parameter configs to sweep.

---

## Suite 1: LISA Comm Cost vs Accuracy Tradeoff

**Hypothesis**: LISA-FedAvg achieves comparable perplexity to plain FedAvg while sending 40–60% fewer gradient tensors.

**Configs to run** (edit `eval/fed_config.py` before each run):

| Run | LISA_MIDDLE_SAMPLE | LISA_BOTTOM | LISA_TOP | Expected comm cost |
|-----|-------------------|-------------|----------|--------------------|
| 1a  | 0 | 2 | 2 | minimum |
| 1b  | 1 | 2 | 2 | low |
| 1c  | 2 | 2 | 2 | baseline (default) |
| 1d  | 3 | 2 | 2 | medium |
| 1e  | all | 2 | 2 | maximum |
| 1f  | 2 | 1 | 1 | aggressive LISA |
| 1g  | 2 | 3 | 3 | full bottom+top |

**Baseline to beat**: Run with `LISA_MIDDLE_SAMPLE = all` (≈ plain FedAvg) as the upper-bound perplexity.

**What to measure**: Perplexity at round 5, total gradient tensors sent per round.

---

## Suite 2: Non-IID Data Skew

**Hypothesis**: LISA layer selection is robust to non-IID (non-independent, non-identically distributed) data — a real-world scenario where clients have very different datasets.

**Setup**: Modify `eval/fedavg_vs_lisafedavg.py` to partition wikitext-2 by topic clusters (e.g., client 0 gets samples 0–400, client 1 gets 400–800, client 2 gets 800–1200) instead of IID random split.

| Run | Data distribution | LISA_MIDDLE_SAMPLE | Notes |
|-----|------------------|-------------------|-------|
| 2a  | IID (current) | 2 | Baseline |
| 2b  | Non-IID by slice | 2 | Skewed labels |
| 2c  | Non-IID by slice | all | FedAvg equivalent |
| 2d  | Non-IID by slice | 1 | LISA with max compression |

**What to measure**: Perplexity divergence between IID and non-IID runs. Large gap = LISA is sensitive to data skew.

---

## Suite 3: Byzantine Resilience Under Attack

**Hypothesis**: Byzantine defenses (Krum, Trimmed Mean, norm-based) maintain accuracy when some clients send malicious updates.

**Setup**: Modify `eval/fedavg_vs_lisafedavg.py` to inject adversarial gradients from 1–2 "malicious" clients per round:
- **Label flip attack**: send gradients with random noise scaled to 10× the true gradient norm
- **Zero gradient attack**: send all-zero gradients

| Run | Method | Malicious clients | Attack type |
|-----|--------|------------------|-------------|
| 3a  | none (plain FedAvg) | 0 | — |
| 3b  | none | 1 | label flip |
| 3c  | krum | 1 | label flip |
| 3d  | trimmed_mean | 1 | label flip |
| 3e  | norm | 1 | label flip |
| 3f  | krum | 2 | label flip |
| 3g  | trimmed_mean | 2 | label flip |
| 3h  | norm | 2 | zero gradient |

**What to measure**: Perplexity degradation under attack. Defended runs should be stable; un-defended should diverge.

---

## Suite 4: Differential Privacy Sweep

**Hypothesis**: Differential privacy (Gaussian noise) provides formal privacy guarantees with controllable utility tradeoff.

**Setup**: Vary `NOISE_MULTIPLIER` in `eval/fed_config.py`. Noise is added per gradient update before sending to server.

| Run | NOISE_MULTIPLIER | Target epsilon (approx) |
|-----|-----------------|----------------------|
| 4a  | 0.0 | ∞ (no privacy) |
| 4b  | 0.1 | ~10 |
| 4c  | 0.5 | ~5 |
| 4d  | 1.0 | ~2–3 |
| 4e  | 2.0 | ~1 |

**What to measure**: Perplexity vs epsilon. Plot privacy-utility tradeoff curve.

---

## Suite 5: Scaling Study

**Hypothesis**: LISA's benefits (comm reduction) hold across model sizes and client counts.

| Run | Model | NUM_CLIENTS | ROUNDS | LISA_MIDDLE_SAMPLE |
|-----|-------|-------------|--------|--------------------|
| 5a  | Pythia-70m | 3 | 5 | 2 |
| 5b  | Pythia-70m | 5 | 5 | 2 |
| 5c  | Pythia-70m | 3 | 10 | 2 |
| 5d  | Pythia-70m | 3 | 5 | 1 |
| 5e  | TinyLlama-1.1B* | 3 | 3 | 2 |

*TinyLlama runs are slow (~51s/step on this machine). Keep ROUNDS low or use LOCAL_STEPS=3.

---

# How to Run Overnight

```bash
# Run all suites in order (each run saves to eval/results/exp_*.json)
# Estimated total: ~3–4 hours for suites 1–4 on Pythia-70m

# Suite 1 — LISA sweep
# Edit fed_config.py, then:
python eval/autora.py run

# Suite 2 — Non-IID (edit fedavg_vs_lisafedavg.py to add skew)
# Suite 3 — Byzantine (edit to inject attacks)
# Suite 4 — DP sweep (edit NOISE_MULTIPLIER)
# Suite 5 — Scaling

# After all runs:
python eval/autora.py best
```

Use `python eval/autora.py suggest` after each suite for AI-generated next steps.

---

# What You Can Tune

In `eval/fed_config.py`:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| LISA_MIDDLE_SAMPLE | 2 | 0 – all | Higher = more comm, better diversity |
| LISA_BOTTOM_LAYERS | 2 | 1–4 | Higher = more comm |
| LISA_TOP_LAYERS | 2 | 1–4 | Higher = more comm |
| LR | 3e-4 | 1e-4 – 1e-3 | Too high = diverge; too low = no learning |
| LOCAL_STEPS | 5 | 1–20 | More = more local compute, less comm |
| NUM_CLIENTS | 3 | 2–10 | More = more comm per round |
| ROUNDS | 5 | 3–20 | More = more comm, potentially better model |
| COMPRESSION_K | 0.1 | 0.05–0.3 | Lower = more compression, risk of quality loss |
| NOISE_MULTIPLIER | 0.0 | 0.0–2.0 | Higher = more privacy, worse perplexity |

---

# Reading Results

Results saved to `eval/results/exp_*.json`. Run:
```bash
python eval/autora.py best    # show best perplexity so far
python eval/autora.py suggest  # get AI advice on next config
```

Key fields per experiment:
- `final_perplexity` — lower is better
- `avg_gradients_per_round` — comm cost per round
- `layer_selection_counts` — which layers LISA picked most often
- `config` — full configuration used
