"""
Decision layer + audit trail for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Turns a bare risk score into a bounded, explainable action:
  allow  -> below the cost-optimal threshold, process normally
  review -> moderate risk, flag for step-up authentication
  block  -> high risk, hold the transaction

Every decision is logged with its reasoning (top SHAP factors) so the
system is auditable — this is the "gated and logged" requirement the
track's evaluation bar calls out explicitly.
"""

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from shap_explainer import build_explainer, explain_alert


# ---------------------------------------------------------------------
# Bounded decision rules — thresholds are the ONLY thing that decides
# an action. No free-form model output ever triggers an action directly.
# ---------------------------------------------------------------------
REVIEW_THRESHOLD = 0.05   # matches the cost-optimal threshold from fraud_classifier.py
BLOCK_THRESHOLD = 0.80    # high-confidence fraud — hold outright

# Hard caps, independent of model confidence — these never change per-transaction
MAX_AUTO_BLOCKS_PER_HOUR = 50   # safety valve against a runaway false-positive spike


def decide_action(risk_score: float) -> str:
    if risk_score >= BLOCK_THRESHOLD:
        return "block"
    elif risk_score >= REVIEW_THRESHOLD:
        return "review"
    else:
        return "allow"


# ---------------------------------------------------------------------
# Safety Cap Manager — thread-safe sliding 1-hour auto-block limiter
# ---------------------------------------------------------------------
class SafetyCapManager:
    """
    Stateful safety valve against runaway false-positive block spikes.
    Tracks block events in a rolling 1-hour window (3600s) and degrades
    subsequent blocks to 'review' once MAX_AUTO_BLOCKS_PER_HOUR is reached.
    """
    def __init__(self, max_blocks_per_hour: int = MAX_AUTO_BLOCKS_PER_HOUR, window_sec: float = 3600.0):
        self.max_blocks_per_hour = max_blocks_per_hour
        self.window_sec = window_sec
        self.block_timestamps: deque = deque()
        self.lock = threading.Lock()

    def evaluate_block(self, now: Optional[float] = None) -> Tuple[str, bool]:
        """
        Evaluates a potential 'block' action against the safety cap.
        Returns:
            (action, capped_by_safety_limit)
            - If under cap: ('block', False)
            - If cap reached: ('review', True)
        """
        if now is None:
            now = time.time()

        with self.lock:
            cutoff = now - self.window_sec
            while self.block_timestamps and self.block_timestamps[0] < cutoff:
                self.block_timestamps.popleft()

            if len(self.block_timestamps) >= self.max_blocks_per_hour:
                return "review", True
            else:
                self.block_timestamps.append(now)
                return "block", False

    def reset(self) -> None:
        """Resets block counters (useful for tests or batch simulations)."""
        with self.lock:
            self.block_timestamps.clear()

    @property
    def current_block_count(self) -> int:
        with self.lock:
            now = time.time()
            cutoff = now - self.window_sec
            while self.block_timestamps and self.block_timestamps[0] < cutoff:
                self.block_timestamps.popleft()
            return len(self.block_timestamps)


# ---------------------------------------------------------------------
# Audit Logger — thread-safe append-only JSONL audit writer
# ---------------------------------------------------------------------
class AuditLogger:
    """
    Thread-safe append-only audit logger writing structured JSONL records.
    """
    def __init__(self, path: str = "audit_trail.jsonl"):
        self.path = path
        self.lock = threading.Lock()

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """Appends a single JSONL entry to the audit trail."""
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def log_batch(self, entries: List[Dict[str, Any]], append: bool = True) -> None:
        """Writes or appends a list of JSONL entries."""
        mode = "a" if append else "w"
        with self.lock:
            with open(self.path, mode, encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------
# Process a batch of transactions -> decisions + full audit trail
# ---------------------------------------------------------------------
def process_batch(model, X_test, y_test, explainer, shap_values, txn_ids=None, safety_cap=None):
    probs = model.predict_proba(X_test)[:, 1]
    audit_log = []

    if safety_cap is None:
        safety_cap = SafetyCapManager()
    else:
        safety_cap.reset()

    if txn_ids is None:
        txn_ids = [f"TXN{100000 + i}" for i in range(len(X_test))]

    # Fixed timestamp base for batch simulation so all batch items fall in the same 1-hr window
    batch_time = time.time()

    for i in range(len(X_test)):
        risk_score = float(probs[i])
        action = decide_action(risk_score)

        if action == "block":
            action, capped = safety_cap.evaluate_block(now=batch_time)
        else:
            capped = False

        reasoning = None
        if action in ("review", "block"):
            reasoning = explain_alert(i, model, X_test, explainer, shap_values, top_n=3, verbose=False, risk_score=risk_score)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_id": txn_ids[i],
            "risk_score": round(risk_score, 4),
            "action": action,
            "capped_by_safety_limit": capped,
            "actual_label": int(y_test.iloc[i]) if hasattr(y_test, "iloc") else int(y_test[i]),
            "top_factors": reasoning["top_factors"] if reasoning else None,
        }
        audit_log.append(entry)

    return audit_log


# ---------------------------------------------------------------------
# Save + summarize the audit trail
# ---------------------------------------------------------------------
def save_audit_trail(audit_log, path="audit_trail.jsonl", append=False):
    logger = AuditLogger(path)
    logger.log_batch(audit_log, append=append)
    action_type = "Appended" if append else "Saved"
    print(f"{action_type} {len(audit_log)} audit entries to {path}")


def summarize(audit_log):
    df = pd.DataFrame(audit_log)
    action_counts = df["action"].value_counts()

    print("\nDecision summary:")
    print(action_counts.to_string())

    capped = df["capped_by_safety_limit"].sum()
    if capped:
        print(f"\n{capped} block decisions were downgraded to 'review' "
              f"due to the {MAX_AUTO_BLOCKS_PER_HOUR}/hour safety cap.")

    # Honesty check: how many blocks/reviews were actually correct?
    flagged = df[df["action"].isin(["review", "block"])]
    if len(flagged) > 0:
        true_fraud_caught = flagged["actual_label"].sum()
        print(f"\nOf {len(flagged)} flagged transactions, {true_fraud_caught} "
              f"were actually fraudulent ({true_fraud_caught/len(flagged):.1%} precision on flagged set).")

    missed = df[(df["action"] == "allow") & (df["actual_label"] == 1)]
    print(f"{len(missed)} fraudulent transactions were allowed through (missed).")

    return df


# ---------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Training model and building explainer...")
    model, X_test, y_test, explainer, shap_values = build_explainer()

    print(f"\nApplying decision rules (review >= {REVIEW_THRESHOLD}, block >= {BLOCK_THRESHOLD})...")
    audit_log = process_batch(model, X_test, y_test, explainer, shap_values)

    save_audit_trail(audit_log)
    summary_df = summarize(audit_log)

    print("\nSample audit entries (first blocked transaction):")
    blocked = [e for e in audit_log if e["action"] == "block"]
    if blocked:
        print(json.dumps(blocked[0], indent=2))
