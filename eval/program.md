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

**Primary metric**: Test perplexity (lower is better — means the model understands language better)

**Secondary metric**: Communication cost (fewer gradient tensors sent = less bandwidth used)

The ideal result is: **perplexity as low as possible** while **communication cost as low as possible**. These trade off against each other — you can reduce comm cost by training fewer layers, but that may hurt perplexity.

## Your Task

You are the autonomous experimenter. Your job:

1. Read `eval/results/` to understand what has already been tried
2. Read `eval/fed_config.py` to understand the current configuration
3. Form a hypothesis about what to change (e.g., "reducing LISA_MIDDLE_SAMPLE from 2 to 1 will cut comm cost by ~15% with <3% perplexity increase")
4. Edit `eval/fed_config.py` with your proposed changes
5. Run `python eval/autora.py run` to execute the experiment
6. Review the results — did your change help? Did perplexity improve? Did comm cost change?
7. Update your hypothesis and repeat

## What You Can Tune

In `eval/fed_config.py`, you can adjust:

### LISA Layer Selection
- `LISA_BOTTOM_LAYERS` (default: 2): Number of bottom transformer layers to always train. More = higher comm cost, potentially better perplexity.
- `LISA_TOP_LAYERS` (default: 2): Number of top transformer layers to always train. More = higher comm cost, potentially better perplexity.
- `LISA_MIDDLE_SAMPLE` (default: 2): Number of randomly sampled middle layers to train each round. More = higher comm cost but potentially better gradient diversity.

### Training
- `LR` (default: 3e-4): Learning rate. Higher = faster training but risk of divergence.
- `LOCAL_STEPS` (default: 5): Number of local training steps per round per client. More = better local training but more compute.
- `NUM_CLIENTS` (default: 3): Number of federated clients. More clients = more diverse data but higher comm.
- `ROUNDS` (default: 5): Number of federated rounds. More rounds = more communication but potentially better model.

### Compression
- `COMPRESSION` (default: "both"): Gradient compression strategy.
  - `"none"`: No compression (all gradients sent as float32)
  - `"sparsify"`: Only send top-K% largest gradients by magnitude
  - `"quantize"`: Quantize gradients to N bits (default 8)
  - `"both"`: Apply both sparsification and quantization
- `COMPRESSION_K` (default: 0.1): For sparsification, the fraction of gradients to keep (0.1 = keep top 10%)
- `COMPRESSION_BITS` (default: 8): Bit depth for quantization

### Privacy
- `NOISE_MULTIPLIER` (default: 0.0): Differential privacy noise level. 0 = disabled. Higher = more privacy, potentially worse model quality.

## How to Read Results

Results are saved as JSON files in `eval/results/exp_N.json` where N is the experiment number.

Example result structure:
```json
{
  "exp_id": "exp_001",
  "timestamp": "2026-03-21T20:00:00",
  "config_hash": "abc123",
  "perplexity_per_round": [42.1, 38.5, 35.2, 33.1, 31.8],
  "comm_cost_per_round": [24, 24, 24, 24, 24],
  "final_perplexity": 31.8,
  "total_comm_cost": 120,
  "layer_selection": {"bottom": 2, "top": 2, "middle": 2},
  "config": { ... full config ... }
}
```

Run `python eval/autora.py best` to see the best experiment so far.

Run `python eval/autora.py suggest` to get AI-generated advice on what to try next.

## Time Budget

Each experiment has a **hard 20-minute wall-clock budget**. The config is set so that 3 clients × 5 rounds × Pythia-70m completes in ~20 minutes on a typical machine. If you change NUM_CLIENTS or ROUNDS, keep this budget in mind.

## Example Workflow

```
# 1. Check current best
python eval/autora.py best

# 2. See suggestions
python eval/autora.py suggest

# 3. Edit config
# (edit eval/fed_config.py)

# 4. Run experiment
python eval/autora.py run

# 5. Check results printed to console and in eval/results/exp_N.json
```

## Tips

- Changes that reduce comm cost may increase perplexity. The sweet spot is where perplexity increase is <5% while comm cost drops significantly.
- LISA_MIDDLE_SAMPLE has the biggest impact on comm cost — each middle layer adds 2 gradient tensors (LoRA A and B).
- Lowering LR too much can make training ineffective. 1e-4 to 5e-4 is the safe range.
- COMPRESSION_K = 0.1 (keep 10%) is aggressive — you might try 0.2 for a perplexity/comm tradeoff.
- NUM_CLIENTS = 3 is fast; 5+ clients gives more data diversity but costs more.
