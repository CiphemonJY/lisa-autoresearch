"""
Checkpoint Manager for Federated Learning.

Provides versioned checkpoints with rollback support.
Each checkpoint stores: model state + metadata (round, perplexity, timestamp, etc.)
"""

import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import pickle


class CheckpointManager:
    """
    Manages versioned model checkpoints with rollback support.

    Checkpoint format:
        <checkpoint_dir>/
            round_<N>_v<version>/
                model.pt
                metadata.json
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model_state: Dict[str, torch.Tensor],
        round_num: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint for the given round.

        Args:
            model_state: state_dict of the model
            round_num: federated round number
            metrics: optional metrics dict (perplexity, client_count, etc.)

        Returns:
            checkpoint_id string like "round_5_v1"
        """
        # Determine next version for this round
        existing = self._existing_for_round(round_num)
        version = len(existing) + 1
        checkpoint_id = f"round_{round_num}_v{version}"
        ckpt_path = self.checkpoint_dir / checkpoint_id

        # Reserve the slot by creating the directory
        ckpt_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save model.pt
            model_path = ckpt_path / "model.pt"
            state_cpu = {k: v.cpu() for k, v in model_state.items()}
            torch.save(state_cpu, model_path)

            # Build metadata
            meta = {
                "checkpoint_id": checkpoint_id,
                "round": round_num,
                "version": version,
                "timestamp": time.time(),
                "timestamp_iso": self._iso_now(),
            }
            if metrics:
                meta["perplexity"] = metrics.get("perplexity")
                meta["client_count"] = metrics.get("client_count")
                meta["compression"] = metrics.get("compression", "none")
                meta["dp_enabled"] = metrics.get("dp_enabled", False)
                meta["avg_gradient_norm"] = metrics.get("avg_gradient_norm")
                meta["round_time"] = metrics.get("round_time")
                meta["extra"] = {
                    k: v for k, v in metrics.items()
                    if k not in (
                        "perplexity", "client_count", "compression",
                        "dp_enabled", "avg_gradient_norm", "round_time",
                    )
                }

            # Save metadata.json
            meta_path = ckpt_path / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)

            return checkpoint_id

        except Exception:
            # Clean up partial write
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)
            raise

    def load(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint by ID.

        Returns:
            dict with keys "model" (state_dict) and "metadata"
        """
        if not self._is_valid_id(checkpoint_id):
            raise ValueError(f"Invalid checkpoint ID: {checkpoint_id}")

        ckpt_path = self.checkpoint_dir / checkpoint_id
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        model_path = ckpt_path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pt not found in {checkpoint_id}")

        meta_path = ckpt_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        model_state = torch.load(model_path, map_location="cpu", weights_only=False)

        return {"model": model_state, "metadata": metadata}

    def rollback(self, checkpoint_id: str) -> bool:
        """
        Validate that a checkpoint exists and is loadable.
        Returns True if rollback is possible.
        """
        try:
            data = self.load(checkpoint_id)
            return "model" in data
        except Exception:
            return False

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Return all checkpoints with their metadata, newest first.
        """
        checkpoints = []
        for ckpt_dir in sorted(self.checkpoint_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if not self._is_valid_id(ckpt_dir.name):
                continue
            meta_path = ckpt_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                # Synthesise from directory name
                parts = ckpt_dir.name.replace("round_", "").split("_v")
                try:
                    meta = {"round": int(parts[0]), "version": int(parts[1])}
                except Exception:
                    meta = {}
                meta["checkpoint_id"] = ckpt_dir.name

            checkpoints.append(meta)

        # Sort newest first (round desc, then version desc)
        checkpoints.sort(key=lambda c: (-c.get("round", 0), -c.get("version", 0)))
        return checkpoints

    def prune(self, keep_last_n: int = 5) -> List[str]:
        """
        Delete old checkpoints, keeping only the most recent N per round.

        Returns list of deleted checkpoint IDs.
        """
        all_ckpts = self.list_checkpoints()
        if len(all_ckpts) <= keep_last_n:
            return []

        # Group by round
        by_round: Dict[int, List[Dict]] = {}
        for c in all_ckpts:
            by_round.setdefault(c.get("round", 0), []).append(c)

        to_delete = []
        for round_num, ckpts in by_round.items():
            if len(ckpts) <= keep_last_n:
                continue
            to_delete.extend(ckpts[keep_last_n:])

        deleted = []
        for ckpt_meta in to_delete:
            ckpt_id = ckpt_meta.get("checkpoint_id")
            if not ckpt_id:
                continue
            ckpt_path = self.checkpoint_dir / ckpt_id
            try:
                shutil.rmtree(ckpt_path)
                deleted.append(ckpt_id)
            except Exception as e:
                print(f"Warning: failed to delete {ckpt_id}: {e}")

        return deleted

    def get_latest(self) -> Optional[str]:
        """Return the ID of the most recent checkpoint, or None."""
        all_ckpts = self.list_checkpoints()
        if not all_ckpts:
            return None
        return all_ckpts[0].get("checkpoint_id")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_valid_id(self, name: str) -> bool:
        return bool(__import__("re").match(r"^round_\d+_v\d+$", name))

    def _existing_for_round(self, round_num: int) -> List[Path]:
        pattern = f"round_{round_num}_v*"
        return sorted(self.checkpoint_dir.glob(pattern))

    @staticmethod
    def _iso_now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
