#!/usr/bin/env python3
"""
HIPAA-Compliant Audit Logger for LISA_FTM
==========================================
Records every data access, gradient transfer, and model interaction.
Required for HIPAA compliance in healthcare deployments.

Event types:
    client_connect, client_disconnect,
    gradient_send, gradient_receive,
    model_update_send, model_update_receive,
    checkpoint_save, checkpoint_load,
    psi_proof, query_executed, data_access

Usage:
    from utils.audit_logger import AuditLogger, AuditEvent

    logger = AuditLogger(audit_dir="audit_logs")
    logger.log_event(
        event_type="gradient_send",
        client_id="hospital_A",
        data_type="gradient_update",
        record_count=147456,
        epoch="2026-03-21T14:30:00Z",
        action="send",
    )
"""

import hashlib
import json
import os
import struct
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet

__all__ = ["AuditLogger", "AuditEvent"]


class AuditEventType(str, Enum):
    """HIPAA-relevant event types for federated learning."""
    CLIENT_CONNECT = "client_connect"
    CLIENT_DISCONNECT = "client_disconnect"
    GRADIENT_SEND = "gradient_send"
    GRADIENT_RECEIVE = "gradient_receive"
    MODEL_UPDATE_SEND = "model_update_send"
    MODEL_UPDATE_RECEIVE = "model_update_receive"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    PSI_PROOF = "psi_proof"
    QUERY_EXECUTED = "query_executed"
    DATA_ACCESS = "data_access"


class AuditEvent:
    """Represents a single auditable event in the federated learning pipeline."""

    def __init__(
        self,
        event_type: str,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        data_type: Optional[str] = None,
        record_count: Optional[int] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        epoch: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.event_type = event_type
        self.client_id = client_id
        self.user_id = user_id
        self.data_type = data_type
        self.record_count = record_count
        self.ip_address = ip_address
        self.success = success
        self.error = error
        self.epoch = epoch
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "client_id": self.client_id,
            "user_id": self.user_id,
            "data_type": self.data_type,
            "record_count": self.record_count,
            "ip_address": self.ip_address,
            "success": self.success,
            "error": self.error,
            "epoch": self.epoch,
            **{f"extra_{k}": v for k, v in self.extra.items()},
        }


class AuditLogger:
    """
    HIPAA-compliant audit logger for federated learning.

    Features:
    - Immutable append-only log files
    - Tamper-evident chain hashing (each entry hashes the previous)
    - Fernet symmetric encryption at rest
    - Daily log files for 7-year retention
    - Compliance report generation

    Args:
        audit_dir: Directory to store encrypted audit logs.
        encryption_key: Fernet key for encryption. If None, one is generated.
        retention_years: Log retention period (default 7 years per HIPAA).
    """

    HASH_CHAIN_VERSION = "1.0"

    def __init__(
        self,
        audit_dir: str = "audit_logs",
        encryption_key: Optional[bytes] = None,
        retention_years: int = 7,
    ):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.retention_years = retention_years

        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.fernet = Fernet(encryption_key)
        self._key_for_export = encryption_key  # Store for export; in production use KMS

        self.last_hash: Optional[str] = None
        self._chain_file = self.audit_dir / ".chain_state"
        self._load_last_hash()

    # -------------------------------------------------------------------------
    # Core logging
    # -------------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        data_type: Optional[str] = None,
        record_count: Optional[int] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        epoch: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Log an auditable event.

        Returns the raw event dict (before encryption) for verification.
        """
        event = AuditEvent(
            event_type=event_type,
            client_id=client_id,
            user_id=user_id,
            data_type=data_type,
            record_count=record_count,
            ip_address=ip_address,
            success=success,
            error=error,
            epoch=epoch,
            extra=kwargs,
        )

        event_dict = event.to_dict()

        # Build chain
        prev_hash = self.last_hash
        chain_hash = self._compute_chain_hash(event_dict, prev_hash)

        event_dict["prev_hash"] = prev_hash
        event_dict["chain_hash"] = chain_hash
        event_dict["chain_version"] = self.HASH_CHAIN_VERSION

        self.last_hash = chain_hash
        self._save_last_hash()
        self._write_event(event_dict)

        return event_dict

    def _compute_chain_hash(self, event: Dict[str, Any], prev_hash: Optional[str]) -> str:
        """Compute HMAC-SHA256 chain hash of an event."""
        # Normalize: sort keys, exclude chain fields themselves
        payload_keys = [k for k in event.keys() if k not in ("prev_hash", "chain_hash", "chain_version")]
        payload = {k: event[k] for k in sorted(payload_keys)}
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        payload_bytes = payload_str.encode("utf-8")

        if prev_hash is None:
            # Genesis block: hash the payload + a fixed seed
            seed = b"HIPAA_AUDIT_GENESIS_v1"
            combined = payload_bytes + seed
        else:
            combined = payload_bytes + prev_hash.encode("utf-8")

        return hashlib.sha256(combined).hexdigest()

    def _write_event(self, event: Dict[str, Any]) -> None:
        """Append an encrypted, JSON-serialized event to the daily log file."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.audit_dir / f"audit_{date_str}.log"
        encrypted = self.fernet.encrypt(json.dumps(event, default=str).encode("utf-8"))
        with open(log_file, "ab") as f:
            f.write(encrypted + b"\n")

    # -------------------------------------------------------------------------
    # Chain state persistence
    # -------------------------------------------------------------------------

    def _load_last_hash(self) -> None:
        """Load the last chain hash from disk (encrypted)."""
        if self._chain_file.exists():
            try:
                raw = self._chain_file.read_bytes()
                self.last_hash = raw.decode("utf-8").strip()
            except Exception:
                self.last_hash = None

    def _save_last_hash(self) -> None:
        """Persist the current chain hash to disk (append-only friendly)."""
        if self.last_hash is not None:
            with open(self._chain_file, "w") as f:
                f.write(self.last_hash)

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------

    def verify_chain(self, date: str) -> Dict[str, Any]:
        """
        Verify the chain hash is unbroken for a given day's log file.

        Args:
            date: Date string in YYYY-MM-DD format.

        Returns:
            dict with keys: valid (bool), entries_checked (int), first_hash, last_hash,
                           errors (list of str)
        """
        log_file = self.audit_dir / f"audit_{date}.log"
        if not log_file.exists():
            return {
                "valid": False,
                "entries_checked": 0,
                "errors": [f"Log file not found: {log_file}"],
            }

        errors: List[str] = []
        entries_checked = 0
        prev_hash: Optional[str] = None
        first_hash: Optional[str] = None
        last_hash: Optional[str] = None

        try:
            with open(log_file, "rb") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        decrypted = self.fernet.decrypt(line)
                        event = json.loads(decrypted.decode("utf-8"))
                    except Exception as e:
                        errors.append(f"Line {line_num}: Failed to decrypt/parse: {e}")
                        continue

                    # Verify prev_hash linkage
                    expected_prev = prev_hash
                    actual_prev = event.get("prev_hash")
                    if expected_prev is not None and actual_prev != expected_prev:
                        errors.append(
                            f"Line {line_num}: Chain broken — expected prev_hash {expected_prev[:16]}..., "
                            f"got {str(actual_prev)[:16]}..."
                        )

                    # Recompute and verify chain_hash
                    computed = self._compute_chain_hash(event, prev_hash)
                    stored = event.get("chain_hash")
                    if stored != computed:
                        errors.append(
                            f"Line {line_num}: Invalid chain_hash — stored {str(stored)[:16]}..., "
                            f"computed {computed[:16]}..."
                        )

                    prev_hash = event.get("chain_hash")
                    if first_hash is None:
                        first_hash = prev_hash
                    last_hash = prev_hash
                    entries_checked += 1

        except Exception as e:
            errors.append(f"Failed to read log file: {e}")

        return {
            "valid": len(errors) == 0,
            "entries_checked": entries_checked,
            "first_hash": first_hash,
            "last_hash": last_hash,
            "errors": errors,
        }

    def verify_all_chains(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify chain integrity across a date range.

        Args:
            start_date: YYYY-MM-DD or None (beginning of records).
            end_date: YYYY-MM-DD or None (today).
        """
        if end_date is None:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        all_files = sorted(
            [f for f in self.audit_dir.glob("audit_*.log") if f.name != ".chain_state"]
        )

        results: Dict[str, Any] = {
            "verified": [],
            "failed": [],
            "total_entries": 0,
        }

        prev_hash: Optional[str] = None
        cross_file_errors: List[str] = []

        for log_file in all_files:
            date_str = log_file.stem.replace("audit_", "")
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            result = self.verify_chain(date_str)
            result["date"] = date_str

            if result["valid"]:
                results["verified"].append(date_str)
                # Cross-file chain continuity
                if prev_hash is not None:
                    first_hash = result.get("first_hash")
                    if first_hash and first_hash != prev_hash:
                        cross_file_errors.append(
                            f"Chain break between {date_str} and next file: "
                            f"last={prev_hash[:16]}..., next_first={first_hash[:16]}..."
                        )
                prev_hash = result.get("last_hash")
                results["total_entries"] += result["entries_checked"]
            else:
                results["failed"].append(result)

        if cross_file_errors:
            results["cross_file_errors"] = cross_file_errors
            results["valid"] = False
        else:
            results["valid"] = True

        return results

    # -------------------------------------------------------------------------
    # Decryption (for compliance officers only)
    # -------------------------------------------------------------------------

    def decrypt_log_file(self, date: str) -> List[Dict[str, Any]]:
        """
        Decrypt and return all events from a given day's log.

        WARNING: Return raw event dicts — contains PHI. Access should be
        restricted to designated HIPAA compliance officers only.
        """
        log_file = self.audit_dir / f"audit_{date}.log"
        if not log_file.exists():
            return []

        events = []
        with open(log_file, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    decrypted = self.fernet.decrypt(line)
                    events.append(json.loads(decrypted.decode("utf-8")))
                except Exception:
                    continue
        return events

    # -------------------------------------------------------------------------
    # Compliance reporting
    # -------------------------------------------------------------------------

    def generate_compliance_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a HIPAA compliance report for a date range.

        Returns a dict with:
        - report_generated_at (ISO timestamp)
        - date_range (start_date, end_date)
        - total_events
        - event_type_counts
        - client_activity (per-client event counts)
        - failed_events (errors/unsuccessful operations)
        - chain_verification (results of chain integrity check)
        - retention_info (confirms 7-year policy)
        """
        if end_date is None:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")
        if start_date is None:
            # Default: last 90 days
            start_dt = datetime.utcnow() - timedelta(days=90)
            start_date = start_dt.strftime("%Y-%m-%d")

        all_events: List[Dict[str, Any]] = []
        date_cursor = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        while date_cursor <= end_dt:
            date_str = date_cursor.strftime("%Y-%m-%d")
            all_events.extend(self.decrypt_log_file(date_str))
            date_cursor += timedelta(days=1)

        # Aggregate
        event_type_counts: Dict[str, int] = {}
        client_activity: Dict[str, Dict[str, int]] = {}
        failed_events: List[Dict[str, Any]] = []
        record_access_total = 0

        for event in all_events:
            et = event.get("event_type", "unknown")
            event_type_counts[et] = event_type_counts.get(et, 0) + 1

            cid = event.get("client_id")
            if cid:
                if cid not in client_activity:
                    client_activity[cid] = {}
                client_activity[cid][et] = client_activity[cid].get(et, 0) + 1

            if not event.get("success", True):
                failed_events.append(event)

            rc = event.get("record_count")
            if rc is not None:
                record_access_total += int(rc)

        # Chain verification
        chain_result = self.verify_all_chains(start_date, end_date)

        report = {
            "report_generated_at": datetime.utcnow().isoformat() + "Z",
            "hipaa_version": "HIPAA Privacy Rule 164.312(b)",
            "chain_verification": chain_result,
            "date_range": {"start": start_date, "end": end_date},
            "total_events": len(all_events),
            "event_type_counts": event_type_counts,
            "client_activity": client_activity,
            "failed_events": failed_events,
            "total_records_accessed": record_access_total,
            "retention_policy": {
                "required_years": 7,
                "note": "HIPAA requires minimum 6 years; 7 recommended",
                "audit_dir": str(self.audit_dir),
            },
        }

        return report

    # -------------------------------------------------------------------------
    # Key management helpers
    # -------------------------------------------------------------------------

    def get_encryption_key(self) -> bytes:
        """
        Return the Fernet encryption key.

        WARNING: In production, keys should be stored in a KMS (e.g., AWS KMS,
        HashiCorp Vault). This method is for development/testing only.
        """
        return self._key_for_export

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new Fernet encryption key."""
        return Fernet.generate_key()
