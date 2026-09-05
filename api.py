"""
VIGIL — Behavioral AI Risk Manager API
"Detect. Explain. Decide. Protect."

FastAPI service wrapping the VIGIL RiskPipeline.
POST a transaction's features, get back a risk score, the bounded action
(allow/review/block), and the top SHAP factors behind the decision.
This turns the model into a resilient real-time risk decisioning service.

Run with:  uvicorn api:app --reload
Then POST to http://127.0.0.1:8000/score
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risk_pipeline import get_pipeline


app = FastAPI(
    title="VIGIL — Behavioral AI Risk Manager API",
    description="Detect. Explain. Decide. Protect. | Real-time behavioral fraud and account-takeover risk detection API.",
)

START_TIME = time.time()

# Shared pipeline initialized once at startup
print("Loading risk pipeline at startup...")
_pipeline = get_pipeline()
print("Pipeline ready.")

ENABLE_DEV_ENDPOINTS = os.getenv("ENABLE_DEV_ENDPOINTS", "true").lower() in ("1", "true", "yes")


class Transaction(BaseModel):
    user_id: int = Field(default=0, ge=0, json_schema_extra={"example": 42})
    amount: float = Field(..., gt=0.0, description="Transaction amount in INR; must be positive.", json_schema_extra={"example": 4200.0})
    amount_zscore: Optional[float] = Field(default=None, json_schema_extra={"example": 3.8})
    time_since_last_txn_min: float = Field(..., ge=0.0, description="Minutes elapsed since prior transaction.", json_schema_extra={"example": 4.5})
    txn_velocity_10min: int = Field(..., ge=0, description="Count of transactions in preceding 10 minutes.", json_schema_extra={"example": 3})
    device_change: int = Field(..., ge=0, le=1, description="Binary flag indicating hardware/fingerprint change.", json_schema_extra={"example": 1})
    geo_dist_from_usual_km: float = Field(..., ge=0.0, description="Kilometers distant from typical user locus.", json_schema_extra={"example": 210.0})
    login_burst_count: int = Field(..., ge=0, description="Count of login events in the last 5 minutes.", json_schema_extra={"example": 2})
    ip_risk_score: float = Field(..., ge=0.0, le=1.0, description="Reputation risk score of client IP in [0.0, 1.0].", json_schema_extra={"example": 0.71})
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction origination in [0, 23].", json_schema_extra={"example": 2})


class ScoreResponse(BaseModel):
    risk_score: float
    action: str
    capped_by_safety_limit: bool = False
    top_factors: Optional[list] = None
    explanation_degraded: bool = False


@app.get("/")
def root():
    return {
        "service": "Fraud-spike detector",
        "system": "VIGIL — Behavioral AI Risk Manager",
        "tagline": "Detect. Explain. Decide. Protect.",
        "status": "ready",
        "endpoints": {
            "/score": "POST a transaction to get a risk score and action",
            "/health": "GET operational and model health status",
            "/metrics": "GET in-process decision and safety cap telemetry",
            "/safety-cap/reset": "POST to reset hourly safety cap (Dev/Admin operation)",
        },
    }


@app.get("/health")
def health_check():
    """
    Lightweight health check reporting service status, uptime, and loaded components.
    Performs zero expensive computation or model retraining.
    """
    uptime = round(time.time() - START_TIME, 2)
    artifacts_on_disk = os.path.exists("artifacts/model.json") and os.path.exists("artifacts/user_baselines.json")
    model_loaded = hasattr(_pipeline, "model") and _pipeline.model is not None
    profiler_loaded = hasattr(_pipeline, "profiler") and _pipeline.profiler is not None and _pipeline.profiler.is_fitted

    status = "healthy" if (model_loaded and profiler_loaded) else "degraded"
    return {
        "status": status,
        "service": "Fraud-spike detector API",
        "uptime_seconds": uptime,
        "model_loaded": model_loaded,
        "profiler_loaded": profiler_loaded,
        "artifacts_on_disk": artifacts_on_disk,
        "model_version": "1.0.0-xgb",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
def get_metrics():
    """
    Operational metrics reporting real-time request counts, decision breakdowns,
    safety cap capacity, and reliability statistics.
    """
    stats = _pipeline.telemetry.get_stats()
    current_blocks = _pipeline.safety_cap.current_block_count
    max_blocks = _pipeline.safety_cap.max_blocks_per_hour
    remaining_capacity = max(0, max_blocks - current_blocks)
    uptime = round(time.time() - START_TIME, 2)

    return {
        "uptime_seconds": uptime,
        "total_requests": stats["total_requests"],
        "decisions": {
            "allow": stats["allow_count"],
            "review": stats["review_count"],
            "block": stats["block_count"],
        },
        "safety_cap": {
            "max_blocks_per_hour": max_blocks,
            "current_hourly_blocks": current_blocks,
            "remaining_block_capacity": remaining_capacity,
            "cap_exhausted": current_blocks >= max_blocks,
        },
        "reliability": {
            "degraded_explanations": stats["degraded_explanation_count"],
            "audit_failures": stats["audit_failure_count"],
        },
    }


@app.post("/safety-cap/reset", summary="Reset safety cap counter (Dev/Testing operation)")
def reset_safety_cap():
    """
    Dev/Testing operational hook to reset the 1-hour rolling safety cap.
    Disabled in production when ENABLE_DEV_ENDPOINTS=false.
    """
    if not ENABLE_DEV_ENDPOINTS:
        raise HTTPException(status_code=403, detail="Reset endpoint disabled in production mode")
    _pipeline.safety_cap.reset()
    return {
        "status": "reset",
        "message": "Safety cap reset successfully.",
        "current_hourly_blocks": _pipeline.safety_cap.current_block_count,
        "remaining_block_capacity": _pipeline.safety_cap.max_blocks_per_hour,
    }


@app.post("/score", response_model=ScoreResponse)
def score_transaction(txn: Transaction):
    """
    Scores an incoming transaction, evaluates bounded actions (allow/review/block),
    enforces the rolling 50-block/hr safety cap, computes on-demand SHAP explanations,
    and logs to the audit trail with fail-safe reliability.
    """
    txn_dict = txn.model_dump() if hasattr(txn, "model_dump") else txn.dict()
    result = _pipeline.score_single(txn_dict, record_audit=True)
    return ScoreResponse(
        risk_score=result["risk_score"],
        action=result["action"],
        capped_by_safety_limit=result.get("capped_by_safety_limit", False),
        top_factors=result["top_factors"],
        explanation_degraded=result.get("explanation_degraded", False),
    )
