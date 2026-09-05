"""
VIGIL — Behavioral AI Risk Manager
"Detect. Explain. Decide. Protect."

Shared risk decision pipeline for VIGIL.
Unified engine connecting feature transformation (UserProfiler), model inference
(XGBoost), decision gating (bounded thresholds), stateful auto-block safety capping
(SafetyCapManager), SHAP local explainability, and append-only audit logging.

Used symmetrically by both FastAPI (for real-time single-transaction scoring)
and Streamlit (for batch simulation and dashboard monitoring).
"""

import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import shap
from xgboost import XGBClassifier

from fraud_classifier import (
    FEATURES,
    load_artifacts,
    export_artifacts,
    generate_synthetic_data,
    train_model,
)
from user_profiler import UserProfiler
from shap_explainer import build_explainer
from decision_layer import (
    decide_action,
    SafetyCapManager,
    AuditLogger,
    REVIEW_THRESHOLD,
    BLOCK_THRESHOLD,
    MAX_AUTO_BLOCKS_PER_HOUR,
    process_batch,
)


import logging
import threading

logger = logging.getLogger("risk_pipeline")


class TelemetryTracker:
    """
    Thread-safe in-process telemetry tracker for operational metrics.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.allow_count = 0
        self.review_count = 0
        self.block_count = 0
        self.degraded_explanation_count = 0
        self.audit_failure_count = 0

    def record_decision(self, action: str, explanation_degraded: bool = False, audit_failed: bool = False):
        with self._lock:
            self.total_requests += 1
            if action == "allow":
                self.allow_count += 1
            elif action == "review":
                self.review_count += 1
            elif action == "block":
                self.block_count += 1
            if explanation_degraded:
                self.degraded_explanation_count += 1
            if audit_failed:
                self.audit_failure_count += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "allow_count": self.allow_count,
                "review_count": self.review_count,
                "block_count": self.block_count,
                "degraded_explanation_count": self.degraded_explanation_count,
                "audit_failure_count": self.audit_failure_count,
            }

    def reset(self):
        with self._lock:
            self.total_requests = 0
            self.allow_count = 0
            self.review_count = 0
            self.block_count = 0
            self.degraded_explanation_count = 0
            self.audit_failure_count = 0


class RiskPipeline:
    """
    Shared, production-aligned risk scoring and decision pipeline.
    """

    def __init__(
        self,
        model: Any,
        profiler: UserProfiler,
        explainer: Any,
        safety_cap: Optional[SafetyCapManager] = None,
        audit_path: str = "audit_trail.jsonl",
        telemetry: Optional[TelemetryTracker] = None,
    ):
        self.model = model
        self.profiler = profiler
        self.explainer = explainer
        self.safety_cap = safety_cap if safety_cap is not None else SafetyCapManager()
        self.audit_logger = AuditLogger(audit_path)
        self.telemetry = telemetry if telemetry is not None else TelemetryTracker()

    def score_single(
        self,
        txn_dict: Dict[str, Any],
        record_audit: bool = True,
    ) -> Dict[str, Any]:
        """
        Scores a single transaction, applies bounded decisions and the safety cap,
        generates SHAP factors on review/block, and appends to the audit trail.
        Features fail-safe degradation: SHAP or audit logging failures never crash scoring.
        """
        txn_data = dict(txn_dict)

        # 1. Feature transformation via UserProfiler
        if txn_data.get("amount_zscore") is None:
            txn_data["amount_zscore"] = self.profiler.transform_single(txn_data)

        # 2. Vectorized 1-row feature DataFrame
        row = pd.DataFrame([txn_data])[FEATURES]

        # 3. Model inference
        risk_score = float(self.model.predict_proba(row)[0, 1])

        # 4. Decision thresholding
        action = decide_action(risk_score)

        # 5. Safety cap enforcement
        if action == "block":
            action, capped = self.safety_cap.evaluate_block()
        else:
            capped = False

        # 6. SHAP explanation for flagged alerts (review or block) — fail-safe
        top_factors = None
        explanation_degraded = False
        if action in ("review", "block"):
            try:
                explainer_result = self.explainer(row)
                contributions = pd.Series(explainer_result.values[0], index=FEATURES)
                top = contributions.abs().sort_values(ascending=False).head(3)
                top_factors = [
                    {
                        "feature": feat,
                        "value": round(float(row[feat].iloc[0]), 3),
                        "contribution": round(float(contributions[feat]), 4),
                        "direction": "increased" if contributions[feat] > 0 else "decreased",
                    }
                    for feat in top.index
                ]
            except Exception as exc:
                logger.warning(f"SHAP explanation generation failed: {exc}. Gracefully returning degraded explanation.")
                explanation_degraded = True
                top_factors = [
                    {
                        "feature": "explanation_unavailable",
                        "value": 0.0,
                        "contribution": 0.0,
                        "direction": "degraded",
                    }
                ]

        # 7. Audit trail record — fail-safe
        txn_id = str(txn_data.get("transaction_id") or f"TXN{int(time.time() * 1000) % 1000000:06d}")
        actual_label = int(txn_data["actual_label"]) if "actual_label" in txn_data and txn_data["actual_label"] is not None else None

        audit_failed = False
        if record_audit:
            try:
                audit_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "transaction_id": txn_id,
                    "risk_score": round(risk_score, 4),
                    "action": action,
                    "capped_by_safety_limit": capped,
                    "actual_label": actual_label,
                    "top_factors": top_factors,
                    "explanation_degraded": explanation_degraded,
                }
                self.audit_logger.log_entry(audit_entry)
            except Exception as exc:
                logger.error(f"Audit logging I/O failure for transaction {txn_id}: {exc}")
                audit_failed = True

        # Update in-process telemetry counters
        self.telemetry.record_decision(
            action=action,
            explanation_degraded=explanation_degraded,
            audit_failed=audit_failed,
        )

        result = {
            "transaction_id": txn_id,
            "risk_score": round(risk_score, 4),
            "action": action,
            "capped_by_safety_limit": capped,
            "top_factors": top_factors,
            "explanation_degraded": explanation_degraded,
            "audit_failed": audit_failed,
            "actual_label": actual_label,
        }

        return result

    def score_batch(
        self,
        df: pd.DataFrame,
        record_audit: bool = True,
        reset_safety_cap: bool = True,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Scores a batch of transactions using vectorized prediction, enforces
        safety cap and bounded rules, computes explanations, and appends to audit.
        """
        df_eval = df.copy()
        if "amount_zscore" not in df_eval.columns:
            df_eval = self.profiler.transform(df_eval)

        X_eval = df_eval[FEATURES]
        y_eval = df_eval["is_fraud"] if "is_fraud" in df_eval.columns else pd.Series([0] * len(df_eval))

        if reset_safety_cap:
            self.safety_cap.reset()

        # Generate batch audit entries
        audit_log = process_batch(
            self.model,
            X_eval,
            y_eval,
            self.explainer,
            shap_values=None,
            safety_cap=self.safety_cap,
        )

        if record_audit:
            self.audit_logger.log_batch(audit_log, append=True)

        res_df = pd.DataFrame(audit_log)
        return res_df, audit_log


# ---------------------------------------------------------------------
# Global singleton instance for in-process sharing
# ---------------------------------------------------------------------
_SHARED_PIPELINE: Optional[RiskPipeline] = None


def get_pipeline(artifact_dir: str = "artifacts", audit_path: str = "audit_trail.jsonl") -> RiskPipeline:
    """
    Returns the shared RiskPipeline instance, loading pre-trained artifacts from disk
    if present to ensure sub-second cold start (< 200ms) without model retraining.
    """
    global _SHARED_PIPELINE
    if _SHARED_PIPELINE is None:
        model_path = os.path.join(artifact_dir, "model.json")
        baselines_path = os.path.join(artifact_dir, "user_baselines.json")

        if os.path.exists(model_path) and os.path.exists(baselines_path):
            model, profiler = load_artifacts(artifact_dir=artifact_dir)
        else:
            df = generate_synthetic_data()
            model, X_test, y_test = train_model(df)
            profiler = getattr(model, "profiler", None)
            if profiler is None:
                profiler = UserProfiler().fit(df)
            export_artifacts(model, profiler, artifact_dir=artifact_dir)

        explainer = shap.TreeExplainer(model)

        _SHARED_PIPELINE = RiskPipeline(
            model=model,
            profiler=profiler,
            explainer=explainer,
            safety_cap=SafetyCapManager(),
            audit_path=audit_path,
        )
    return _SHARED_PIPELINE
