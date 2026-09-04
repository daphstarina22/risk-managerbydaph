"""
FastAPI service wrapping the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

POST a transaction's features, get back a risk score, the bounded action
(allow/review/block), and the top SHAP factors behind the decision.
This is what turns the model from "a script that runs" into "a service
that could plausibly sit behind a real payment flow."

Run with:  uvicorn api:app --reload
Then POST to http://127.0.0.1:8000/score
"""

from typing import Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from fraud_classifier import FEATURES
from shap_explainer import build_explainer, explain_alert
from decision_layer import decide_action


app = FastAPI(
    title="Fraud-spike detector API",
    description="Razorpay Buildathon — AI Risk Manager track",
)

# Train once at startup, not per-request — a real deployment would load a
# saved model artifact instead, but this keeps the demo self-contained.
print("Loading model at startup...")
_model, _X_test, _y_test, _explainer, _shap_values = build_explainer()
print("Model ready.")


class Transaction(BaseModel):
    amount: float = Field(..., json_schema_extra={"example": 4200.0})
    amount_zscore: float = Field(..., json_schema_extra={"example": 3.8})
    time_since_last_txn_min: float = Field(..., json_schema_extra={"example": 4.5})
    txn_velocity_10min: int = Field(..., json_schema_extra={"example": 3})
    device_change: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    geo_dist_from_usual_km: float = Field(..., json_schema_extra={"example": 210.0})
    login_burst_count: int = Field(..., json_schema_extra={"example": 2})
    ip_risk_score: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.71})
    hour_of_day: int = Field(..., ge=0, le=23, json_schema_extra={"example": 2})


class ScoreResponse(BaseModel):
    risk_score: float
    action: str
    top_factors: Optional[list] = None


@app.get("/")
def root():
    return {
        "service": "Fraud-spike detector",
        "status": "ready",
        "endpoints": {"/score": "POST a transaction to get a risk score and action"},
    }


@app.post("/score", response_model=ScoreResponse)
def score_transaction(txn: Transaction):
    row = pd.DataFrame([txn.dict()])[FEATURES]

    risk_score = float(_model.predict_proba(row)[0, 1])
    action = decide_action(risk_score)

    top_factors = None
    if action in ("review", "block"):
        # Explain against this single row by temporarily treating it as index 0
        # of a one-row batch — reuses the same SHAP machinery as the batch path.
        explainer_result = _explainer(row)
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

    return ScoreResponse(risk_score=round(risk_score, 4), action=action, top_factors=top_factors)
