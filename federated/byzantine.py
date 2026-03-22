#!/usr/bin/env python3
"""
Byzantine-Resilient Gradient Aggregation for Federated Learning

Provides three methods to defend against malicious client updates:
1. Krum / Multi-Krum  - robust geometric selection
2. Trimmed Mean        - coordinate-wise statistical trimming
3. Norm-based detection - heuristic outlier detection

Reference: Blanchard et al., "Machine Learning with Adversaries: Byzantine
Tolerant Gradient Aggregation" (NeurIPS 2017).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("byzantine")


# ============================================================================
# Helper: flatten gradient dicts into a single vector per client
# ============================================================================

def _grad_dict_to_vector(grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate all gradient tensors in a dict into one flat vector."""
    vecs = []
    for v in grad_dict.values():
        vecs.append(v.flatten().float())
    return torch.cat(vecs)


def _vector_to_grad_dict(
    vector: torch.Tensor,
    template: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Split a flat vector back into a dict of original shapes."""
    result = {}
    offset = 0
    for name, tensor in template.items():
        numel = tensor.numel()
        chunk = vector[offset : offset + numel]
        result[name] = chunk.reshape(tensor.shape).type_as(tensor)
        offset += numel
    return result


# ============================================================================
# Method 1: Krum / Multi-Krum
# ============================================================================

def krum_select(
    grad_dicts: List[Dict[str, torch.Tensor]],
    f: int,
    client_weights: Optional[List[float]] = None,
    multi: bool = True,
) -> Tuple[List[int], Dict]:
    """
    Select honest gradient(s) using Krum or Multi-Krum.

    Krum score for client i:
        score_i = sum_{j ∈ knn(i)} ||g_i - g_j||²
    where knn(i) = nearest n-f-2 neighbors of i (excluding i itself).

    Multi-Krum: average the n-f-1 gradients with the lowest scores.
    Single Krum: return only the gradient with the lowest score.

    Args:
        grad_dicts: list of {param_name: tensor} gradient dicts (one per client)
        f: max number of malicious clients
        client_weights: optional weights (unused by Krum itself, returned in stats)
        multi: if True return n-f-1 selected indices; if False return 1

    Returns:
        (selected_indices, stats_dict)

    Note on n, f:
        We need n >= 2f+3 for Krum to guarantee correctness.
        If fewer clients, we fall back to all-but-worst-(f).
    """
    n = len(grad_dicts)
    stats = {"method": "multi-krum" if multi else "krum", "n": n, "f": f}

    if n == 0:
        return [], stats
    if n == 1:
        return [0], stats

    # Minimum viable n for Krum: need at least 2f+3 to have n-f-2 >= f+1 neighbors
    min_viable_n = 2 * f + 3
    use_simple_fallback = n < min_viable_n

    # Convert to vectors
    vectors = [_grad_dict_to_vector(g) for g in grad_dicts]
    template = grad_dicts[0]

    # Pairwise squared distances: O(n²)
    dists = torch.zeros(n, n)
    for i in range(n):
        diffs = vectors[i].unsqueeze(0) - torch.stack(vectors)  # (n, D)
        dists[i] = (diffs ** 2).sum(dim=1)

    # Number of nearest neighbors to consider
    if use_simple_fallback:
        # Fallback: exclude the f clients with largest average distance
        # (still defensible but less optimal)
        logger.warning(
            f"Krum: n={n} < 2f+3={min_viable_n}, using nearest-n-minus-f fallback"
        )
        stats["fallback"] = True
        k = max(1, n - f)
    else:
        k = n - f - 2  # standard Krum: nearest n-f-2

    # Score = sum of squared distances to k nearest neighbors
    scores = []
    for i in range(n):
        # Exclude self (distance 0), find k nearest others
        row = dists[i].clone()
        row[i] = float("inf")
        knn_vals, _ = row.topk(k, largest=False)
        scores.append(knn_vals.sum().item())

    scores = np.array(scores)

    if multi and n >= f + 1:
        # Select the n-f-1 lowest-scoring clients (standard multi-krum)
        num_select = max(1, n - f - 1)
        selected = np.argsort(scores)[:num_select].tolist()
    else:
        # Single Krum
        selected = [int(np.argmin(scores))]

    stats["scores"] = scores.tolist()
    stats["selected"] = selected
    stats["k"] = k

    return selected, stats


def krum_aggregate(
    grad_dicts: List[Dict[str, torch.Tensor]],
    f: int,
    client_weights: Optional[List[float]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Aggregate gradients using Multi-Krum.

    Returns:
        (aggregated_grad_dict, stats)
    """
    n = len(grad_dicts)
    if n == 0:
        return {}, {"method": "krum", "status": "no_gradients"}
    if n == 1:
        return grad_dicts[0], {"method": "krum", "status": "single_client"}

    selected, stats = krum_select(grad_dicts, f, client_weights, multi=True)

    # Simple average of selected gradients
    template = grad_dicts[0]
    aggregated = {}
    for key in template:
        tensors = [grad_dicts[i][key].float() for i in selected]
        stacked = torch.stack(tensors)
        aggregated[key] = stacked.mean(dim=0).type_as(template[key])

    stats["status"] = "success"
    stats["num_selected"] = len(selected)
    stats["num_excluded"] = n - len(selected)

    return aggregated, stats


# ============================================================================
# Method 2: Trimmed Mean
# ============================================================================

def trimmed_mean_aggregate(
    grad_dicts: List[Dict[str, torch.Tensor]],
    alpha: float = 0.1,
    client_weights: Optional[List[float]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Aggregate gradients using coordinate-wise Trimmed Mean.

    For each gradient dimension:
        1. Sort values across all n clients
        2. Remove top alpha*n and bottom alpha*n values
        3. Average the remaining (1 - 2*alpha)*n values

    With weights, take weighted trimmed mean:
        - Sort by value, keep the middle portion, but weighted average
        - Simplified: use weights in the final average of kept clients

    Args:
        grad_dicts: list of gradient dicts
        alpha: fraction to trim from each tail (0.0 to 0.5)
        client_weights: optional list of weights (e.g. sample counts)

    Returns:
        (aggregated_grad_dict, stats)
    """
    n = len(grad_dicts)
    stats = {"method": "trimmed_mean", "alpha": alpha, "n": n}

    if n == 0:
        return {}, {"status": "no_gradients"}
    if n == 1:
        return grad_dicts[0], {"status": "single_client"}

    # Clamp alpha
    max_trim = (n - 1) / 2 / n  # can't trim more than (n-1)/2 elements total
    alpha = min(alpha, max_trim)

    template = grad_dicts[0]
    aggregated = {}

    num_trim = max(0, int(np.floor(alpha * n)))  # trim this many from each tail

    for key in template:
        tensors = [grad_dicts[i][key].float() for i in range(n)]
        # Stack to (n, D) where D = numel
        stacked = torch.stack([t.flatten() for t in tensors], dim=0)  # (n, D)
        D = stacked.shape[1]

        result = torch.zeros(D, dtype=torch.float32)

        # Per-dimension trimmed mean
        for d in range(D):
            col = stacked[:, d]  # (n,)
            sorted_vals, sort_idx = torch.sort(col)
            # Trim num_trim from each end
            trimmed = sorted_vals[num_trim : n - num_trim]
            if client_weights is not None:
                # Weight the kept values proportionally
                kept_indices = sort_idx[num_trim : n - num_trim]
                kept_weights = torch.tensor(
                    [client_weights[i] for i in kept_indices], dtype=torch.float32
                )
                w_sum = kept_weights.sum()
                if w_sum > 0:
                    result[d] = (trimmed * (kept_weights / w_sum)).sum()
                else:
                    result[d] = trimmed.mean()
            else:
                result[d] = trimmed.mean() if len(trimmed) > 0 else sorted_vals.mean()

        aggregated[key] = result.reshape(tensors[0].shape).type_as(tensors[0])

    num_excluded = num_trim * 2
    stats["status"] = "success"
    stats["num_trimmed_per_side"] = num_trim
    stats["num_kept_per_dim"] = n - num_excluded

    return aggregated, stats


# ============================================================================
# Method 3: Norm-Based Detection
# ============================================================================

def norm_based_aggregate(
    grad_dicts: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[float]] = None,
    sigma_threshold: float = 3.0,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Detect and exclude outliers using L2 norm statistics.

    Compute the L2 norm of each client's gradient update.
    Flag as Byzantine any gradient whose norm is > sigma_threshold standard
    deviations from the mean.

    Falls back to plain FedAvg if all clients are flagged (edge case).

    Args:
        grad_dicts: list of gradient dicts
        client_weights: optional weights for weighted average
        sigma_threshold: number of standard deviations for outlier threshold

    Returns:
        (aggregated_grad_dict, stats)
    """
    n = len(grad_dicts)
    stats = {"method": "norm_based", "sigma_threshold": sigma_threshold}

    if n == 0:
        return {}, {"status": "no_gradients"}
    if n == 1:
        return grad_dicts[0], {"method": "norm_based", "status": "single_client"}

    # Compute L2 norm per client
    norms = torch.zeros(n)
    for i, gd in enumerate(grad_dicts):
        vec = _grad_dict_to_vector(gd)
        norms[i] = torch.norm(vec).item()

    mean_norm = norms.mean().item()
    # Use Median Absolute Deviation (MAD) — robust to outliers.
    # MAD = median(|x_i - median(x)|).  Modified z-score = 0.6745*(x-median)/MAD.
    # A value with modified z-score > sigma_threshold is flagged as outlier.
    sorted_norms, _ = torch.sort(norms)
    mid = n // 2
    if n % 2 == 0:
        median_norm = (sorted_norms[mid - 1] + sorted_norms[mid]).item() * 0.5
    else:
        median_norm = sorted_norms[mid].item()
    abs_devs = torch.abs(norms - median_norm)
    sorted_devs, _ = torch.sort(abs_devs)
    mid_d = n // 2
    if n % 2 == 0:
        mad = (sorted_devs[mid_d - 1] + sorted_devs[mid_d]).item() * 0.5
    else:
        mad = sorted_devs[mid_d].item()

    # Modified z-score using constant for normal distribution consistency
    if mad < 1e-10:
        # All norms essentially identical; no meaningful outliers
        threshold = median_norm
        outliers = [False] * n
        modified_z_vals = [0.0] * n
    else:
        # Modified z-score = 0.6745 * |x - median| / MAD
        # Flag gradients with modified z-score > sigma_threshold
        modified_z_vals = (0.6745 * abs_devs / mad).tolist()
        modified_z = 0.6745 * abs_devs / mad
        outliers = (modified_z > sigma_threshold).tolist()
        threshold = median_norm + sigma_threshold * (mad / 0.6745)

    outlier_ids = [i for i in range(n) if outliers[i]]

    stats["norms"] = norms.tolist()
    stats["median_norm"] = median_norm
    stats["mad"] = mad
    stats["modified_z_scores"] = modified_z_vals
    stats["threshold"] = threshold
    stats["outliers"] = outlier_ids
    stats["num_outliers"] = len(outlier_ids)

    if len(outlier_ids) > 0:
        outlier_norms = [norms[i] for i in outlier_ids]
        logger.warning(
            f"Norm-based detection: flagged {len(outlier_ids)}/{n} clients as "
            f"Byzantine (threshold={threshold:.4f}, median={median_norm:.4f}, "
            f"MAD={mad:.4f}): norms={outlier_norms}"
        )

    # Keep only non-outliers
    kept_indices = [i for i in range(n) if not outliers[i]]

    if len(kept_indices) == 0:
        # Edge case: everything looks Byzantine → fall back to plain FedAvg
        logger.warning(
            "Norm-based: ALL clients flagged as outliers. "
            "Falling back to plain FedAvg."
        )
        kept_indices = list(range(n))
        stats["fallback"] = True

    if len(kept_indices) == 1:
        # Single client left → return directly
        return grad_dicts[kept_indices[0]], stats

    # Weighted average of kept gradients
    template = grad_dicts[0]
    aggregated = {}

    if client_weights is not None:
        # Normalize weights for kept clients
        kept_weights = [client_weights[i] for i in kept_indices]
        total_weight = sum(kept_weights)
        norm_weights = [w / total_weight for w in kept_weights]
    else:
        # Equal weights
        norm_weights = [1.0 / len(kept_indices)] * len(kept_indices)

    for key in template:
        accum = torch.zeros_like(template[key], dtype=torch.float32)
        for idx, w in zip(kept_indices, norm_weights):
            accum += grad_dicts[idx][key].float() * w
        aggregated[key] = accum.type_as(template[key])

    stats["status"] = "success"
    stats["num_kept"] = len(kept_indices)
    stats["num_excluded"] = n - len(kept_indices)

    return aggregated, stats


# ============================================================================
# Unified Byzantine-Resilient Aggregator
# ============================================================================

class ByzantineResilientAggregator:
    """
    Main entry point for Byzantine-resilient aggregation.

    Wraps Krum, Trimmed Mean, and norm-based detection behind a
    consistent `.aggregate()` interface.

    Usage::

        byz = ByzantineResilientAggregator(method="krum", f=1, alpha=0.1)
        result = byz.aggregate(
            grad_dicts=[g1, g2, g3],
            client_weights=[100, 100, 100],
        )
        # result is a {param_name: tensor} gradient dict
    """

    def __init__(
        self,
        method: str = "norm",
        f: int = 1,
        alpha: float = 0.1,
        sigma_threshold: float = 3.0,
    ):
        """
        Args:
            method: "krum", "trimmed_mean", or "norm"
            f: max number of expected malicious clients (used by Krum)
            alpha: fraction to trim from each tail for Trimmed Mean (e.g. 0.1 = 10%)
            sigma_threshold: standard deviations from mean for norm-based detection
        """
        valid_methods = {"krum", "trimmed_mean", "norm"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")
        if not (0 <= alpha < 0.5):
            raise ValueError(f"alpha must be in [0, 0.5), got {alpha}")
        if f < 0:
            raise ValueError(f"f must be non-negative, got {f}")

        self.method = method
        self.f = f
        self.alpha = alpha
        self.sigma_threshold = sigma_threshold

    def aggregate(
        self,
        grad_dicts: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Aggregate client gradients using the configured Byzantine-resilient method.

        Args:
            grad_dicts: list of {param_name: tensor} gradient dicts from clients
            client_weights: optional list of weights (e.g. sample counts).
                             If provided, used for weighted aggregation where applicable.

        Returns:
            (aggregated_grad_dict, stats_dict)

            aggregated_grad_dict: {param_name: tensor} — ready to apply to model
            stats_dict: metadata about the aggregation, including which clients
                        were excluded and why
        """
        n = len(grad_dicts)
        if n == 0:
            return {}, {"status": "no_gradients"}

        if n == 1:
            return grad_dicts[0], {"status": "single_client", "method": self.method}

        # Ensure all dicts have the same keys
        first_keys = set(grad_dicts[0].keys())
        for gd in grad_dicts[1:]:
            if set(gd.keys()) != first_keys:
                raise ValueError(
                    f"Gradient dict key mismatch: {first_keys} vs {set(gd.keys())}"
                )

        # Normalise client_weights
        if client_weights is not None:
            if len(client_weights) != n:
                raise ValueError(
                    f"client_weights length {len(client_weights)} != n {n}"
                )
            # Weights must be positive
            client_weights = [max(w, 1e-8) for w in client_weights]
        else:
            client_weights = [1.0] * n

        logger.info(
            f"ByzantineResilientAggregator: method={self.method}, n={n}, f={self.f}"
        )

        if self.method == "krum":
            result, stats = krum_aggregate(
                grad_dicts, f=self.f, client_weights=client_weights
            )
        elif self.method == "trimmed_mean":
            result, stats = trimmed_mean_aggregate(
                grad_dicts, alpha=self.alpha, client_weights=client_weights
            )
        elif self.method == "norm":
            result, stats = norm_based_aggregate(
                grad_dicts,
                client_weights=client_weights,
                sigma_threshold=self.sigma_threshold,
            )
        else:
            raise RuntimeError(f"Unknown method: {self.method}")

        stats["method"] = self.method
        return result, stats
