# ---------------------------------------------------------------------------
# LISA-FTM Autoresearch Experiment Configuration
# ---------------------------------------------------------------------------
# Edit this file to propose new experiments.
# The agent (autora.py run) will execute an experiment using these values.
# ---------------------------------------------------------------------------

# LISA layer selection params
LISA_BOTTOM_LAYERS = 2
LISA_TOP_LAYERS = 2
LISA_MIDDLE_SAMPLE = 2

# Training params
LR = 3e-4
LOCAL_STEPS = 5
NUM_CLIENTS = 3
ROUNDS = 5
MODEL = "EleutherAI/pythia-70m"

# Compression
COMPRESSION = "both"  # none/sparsify/quantize/both
COMPRESSION_K = 0.1
COMPRESSION_BITS = 8

# Differential privacy (0 = disabled)
NOISE_MULTIPLIER = 0.0

# What to optimize
# primary metric: perplexity (lower is better)
# secondary metric: comm_cost (number of gradient tensors sent)
