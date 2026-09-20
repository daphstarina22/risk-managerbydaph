"""
VIGIL — Behavioral AI Risk Manager
Detect • Explain • Decide • Protect

Streamlit Dashboard & Interactive What-If Simulator.
Features:
  1. What-If Fraud Risk Simulator: Interactive transaction feature tweaking with 6 presets,
     dynamic baseline z-score derivation, bounded actions, safety cap, and on-demand TreeSHAP.
  2. Live Risk Simulation: Real-time risk operations console streaming attack scenarios.
  3. Flagged Alert Feed: Batch transaction monitoring with precision/recall metrics and per-alert SHAP.
  4. Audit & Telemetry: Live append-only audit trail and in-process operational metrics.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

from decision_layer import (
    BLOCK_THRESHOLD,
    MAX_AUTO_BLOCKS_PER_HOUR,
    REVIEW_THRESHOLD,
)
from fraud_classifier import FEATURES, generate_synthetic_data
from risk_pipeline import get_pipeline


st.set_page_config(
    page_title="VIGIL — Behavioral AI Risk Manager",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom Dark Theme & Professional Fintech Styles
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark Theme Core Elements */
    .stApp {
        background-color: #080c14;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Main Branding Banner */
    .vigil-banner {
        background: #0d1424;
        border: 1px solid #1c273e;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    /* Compact Header Status Strip */
    .vigil-status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 1.75rem;
        margin-top: 1rem;
        padding-top: 0.85rem;
        border-top: 1px solid #1c273e;
    }
    .status-strip-item {
        display: flex;
        flex-direction: column;
    }
    .status-strip-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: #64748b;
        text-transform: uppercase;
    }
    .status-strip-value {
        font-size: 0.86rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 2px;
    }
    .status-val-active {
        color: #10b981;
    }

    /* 4 Pillars Card Layout */
    .pillar-card {
        background: #0d1424;
        border: 1px solid #1c273e;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        min-height: 100px;
    }
    .pillar-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f8fafc;
        margin-bottom: 0.2rem;
        letter-spacing: 0.3px;
    }
    .pillar-sub {
        font-weight: 600;
        font-size: 0.78rem;
        color: #38bdf8;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pillar-desc {
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.35;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #0d1424;
        border: 1px solid #1c273e;
        border-radius: 6px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700;
        color: #f8fafc;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Context & Storytelling Box */
    .scenario-box {
        background: #0d1424;
        border: 1px solid #1c273e;
        border-radius: 6px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 0.85rem;
    }

    /* Transaction Result Card */
    .result-card {
        background: #0d1424;
        border: 1px solid #1c273e;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }
    .result-card.result-allow {
        border-left: 4px solid #10b981;
    }
    .result-card.result-review {
        border-left: 4px solid #f59e0b;
    }
    .result-card.result-block {
        border-left: 4px solid #ef4444;
    }
    .result-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        border-bottom: 1px solid #1c273e;
        padding-bottom: 0.85rem;
        margin-bottom: 0.85rem;
    }
    .result-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .result-score-num {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.1;
    }
    .badge-allow {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 16px;
        border-radius: 4px;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        display: inline-block;
    }
    .badge-review {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 6px 16px;
        border-radius: 4px;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        display: inline-block;
    }
    .badge-block {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 6px 16px;
        border-radius: 4px;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        display: inline-block;
    }
    .badge-capped {
        background: rgba(139, 92, 246, 0.15);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.4);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 8px;
        display: inline-block;
    }
    .result-rationale-box {
        margin-top: 0.5rem;
    }
    .result-rationale-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .result-rationale-desc {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.45;
    }

    /* Horizontal Risk Score Meter */
    .risk-meter-wrapper {
        background: #0d1424;
        border: 1px solid #1c273e;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .risk-meter-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .risk-meter-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    .risk-meter-current {
        font-size: 0.82rem;
        color: #f1f5f9;
        font-family: ui-monospace, SFMono-Regular, monospace;
    }
    .risk-meter-track-container {
        position: relative;
        height: 28px;
        margin: 0.6rem 0 0.4rem 0;
    }
    .risk-meter-track {
        display: flex;
        height: 100%;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #1c273e;
    }
    .track-zone {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        padding: 0 4px;
    }
    .zone-allow {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border-right: 1px dashed rgba(16, 185, 129, 0.5);
    }
    .zone-review {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border-right: 1px dashed rgba(239, 68, 68, 0.5);
    }
    .zone-block {
        background: rgba(239, 68, 68, 0.20);
        color: #f87171;
    }
    .risk-meter-marker {
        position: absolute;
        top: -4px;
        bottom: -4px;
        width: 0px;
        transform: translateX(-50%);
        pointer-events: none;
    }
    .marker-line {
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        width: 3px;
        background: #ffffff;
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.7);
        border-radius: 2px;
    }
    .marker-pill {
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        background: #ffffff;
        color: #080c14;
        font-size: 0.68rem;
        font-weight: 800;
        padding: 1px 5px;
        border-radius: 3px;
        font-family: ui-monospace, SFMono-Regular, monospace;
        white-space: nowrap;
    }
    .risk-meter-ticks {
        display: flex;
        justify-content: space-between;
        position: relative;
        font-size: 0.68rem;
        color: #64748b;
        font-family: ui-monospace, SFMono-Regular, monospace;
        margin-top: 1.25rem;
    }

    /* Tag Pills */
    .tag-pill {
        display: inline-block;
        background: #131d31;
        color: #94a3b8;
        border: 1px solid #22324e;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Preset Scenarios for What-If Simulator
# ---------------------------------------------------------------------
SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Normal Transaction": {
        "user_id": 42,
        "amount": 450.0,
        "time_since_last_txn_min": 120.0,
        "txn_velocity_10min": 0,
        "device_change": 0,
        "geo_dist_from_usual_km": 1.5,
        "login_burst_count": 0,
        "ip_risk_score": 0.05,
        "hour_of_day": 14,
        "description": "Behavior is consistent with the user's normal profile. ₹450 routine spend from known device at usual location.",
        "context": "Routine daytime transaction from recognized hardware and location with low network risk. No behavioral anomaly.",
        "tags": ["Low Risk", "Known Device", "Baseline Amount", "Domestic IP"],
    },
    "Unusual Amount": {
        "user_id": 0,
        "amount": 18500.0,
        "time_since_last_txn_min": 45.0,
        "txn_velocity_10min": 1,
        "device_change": 0,
        "geo_dist_from_usual_km": 3.0,
        "login_burst_count": 0,
        "ip_risk_score": 0.45,
        "hour_of_day": 16,
        "description": "Transaction amount is significantly higher than the user's usual spending pattern, while other signals remain relatively normal.",
        "context": "High-value purchase (₹18,500) causing significant z-score deviation (+21.1σ), with other signals remaining moderate.",
        "tags": ["Amount Anomaly", "Step-Up Review", "Known Device", "Moderate Network"],
    },
    "New Device": {
        "user_id": 105,
        "amount": 1500.0,
        "time_since_last_txn_min": 90.0,
        "txn_velocity_10min": 1,
        "device_change": 1,
        "geo_dist_from_usual_km": 8.0,
        "login_burst_count": 1,
        "ip_risk_score": 0.35,
        "hour_of_day": 19,
        "description": "A transaction is being made from an unfamiliar device, with other signals remaining moderate.",
        "context": "First-time hardware fingerprint with moderate amount and slight IP elevation. Evaluates device switching behavior.",
        "tags": ["Hardware Change", "Step-Up Review", "Low Ticket", "Evening Spend"],
    },
    "High Velocity Transaction": {
        "user_id": 210,
        "amount": 1200.0,
        "time_since_last_txn_min": 0.8,
        "txn_velocity_10min": 5,
        "device_change": 0,
        "geo_dist_from_usual_km": 2.0,
        "ip_risk_score": 0.20,
        "login_burst_count": 0,
        "hour_of_day": 11,
        "description": "Multiple transactions (5 txns in 10 min) are occurring within a short period of time.",
        "context": "Rapid succession of payments (under 1 minute apart) on a known device, simulating automated card testing or script botting.",
        "tags": ["Velocity Spike", "Rapid Interval", "Known Device", "Review / Hold"],
    },
    "Account Takeover Scenario": {
        "user_id": 315,
        "amount": 12000.0,
        "time_since_last_txn_min": 1.5,
        "txn_velocity_10min": 3,
        "device_change": 1,
        "geo_dist_from_usual_km": 320.0,
        "login_burst_count": 3,
        "ip_risk_score": 0.75,
        "hour_of_day": 2,
        "description": "Several suspicious authentication and behavioral signals occur together.",
        "context": "Multi-signal ATO: 3 burst logins, 320 km location jump, new device fingerprint, and ₹12,000 transfer at 2 AM.",
        "tags": ["Account Takeover", "Geo Velocity Jump", "Off-Hours (2 AM)", "Immediate Block"],
    },
    "High-Risk Combined Attack": {
        "user_id": 450,
        "amount": 35000.0,
        "time_since_last_txn_min": 0.1,
        "txn_velocity_10min": 7,
        "device_change": 1,
        "geo_dist_from_usual_km": 520.0,
        "login_burst_count": 5,
        "ip_risk_score": 0.98,
        "hour_of_day": 3,
        "description": "Multiple high-risk signals are combined, representing a severe attack scenario.",
        "context": "Severe extraction attack: TOR/proxy IP (0.98), 5 burst logins, extreme velocity, 520 km jump, and ₹35,000 top-up at 3 AM.",
        "tags": ["Severe Attack", "High-Risk Proxy", "Rapid Drain", "Critical Block"],
    },
}


# ---------------------------------------------------------------------
# Dynamic Humanized Storytelling Generator
# ---------------------------------------------------------------------
def generate_human_explanation(txn_payload: Dict[str, Any], result: Dict[str, Any], zscore: float) -> Dict[str, str]:
    """
    Generates plain-language executive rationales grounded entirely in the
    actual transaction inputs and model/decision layer outputs.
    """
    action = result["action"]
    risk_score = result["risk_score"]
    capped = result.get("capped_by_safety_limit", False)

    amount = txn_payload["amount"]
    user_id = txn_payload["user_id"]
    hour = txn_payload["hour_of_day"]
    geo_dist = txn_payload["geo_dist_from_usual_km"]
    velocity = txn_payload["txn_velocity_10min"]
    device_change = txn_payload["device_change"]
    login_burst = txn_payload["login_burst_count"]
    ip_risk = txn_payload["ip_risk_score"]

    # 1. WHAT HAPPENED?
    dev_str = "unfamiliar hardware" if device_change else "recognized device"
    what_happened = (
        f"User #{user_id} initiated a payment of ₹{amount:,.2f} at {hour:02d}:00 hours "
        f"from an {dev_str}."
    )

    # 2. WHAT DID VIGIL NOTICE?
    notices = []
    if zscore >= 3.0:
        notices.append(f"Significant spend anomaly ({zscore:+.1f}σ deviation from user baseline)")
    elif zscore >= 1.5:
        notices.append(f"Elevated spend amount ({zscore:+.1f}σ above typical spend)")

    if geo_dist >= 100.0:
        notices.append(f"Geographic jump ({geo_dist:,.0f} km distant from habitual location)")
    if velocity >= 3:
        notices.append(f"High velocity burst ({velocity} transactions within 10 minutes)")
    if device_change == 1:
        notices.append("New device fingerprint not previously associated with account")
    if login_burst >= 2:
        notices.append(f"Authentication anomaly ({login_burst} logins in preceding 5 minutes)")
    if ip_risk >= 0.50:
        notices.append(f"Suspicious network threat index ({ip_risk:.2f} external risk score)")
    if hour in (1, 2, 3, 4):
        notices.append(f"High-risk nighttime window ({hour:02d}:00 local time)")

    if not notices:
        notices_text = "All behavioral signals remain well within historical tolerances and baseline limits."
    else:
        notices_text = " • " + "\n • ".join(notices)

    # 3. WHAT DID VIGIL DECIDE? & 4. WHY?
    if action == "allow":
        what_decided = f"ALLOW (Approved) — Risk score {risk_score:.4f} is below the {REVIEW_THRESHOLD} threshold."
        why = (
            "This transaction is consistent with the user's normal activity. "
            "No significant combination of risk signals was detected, allowing the payment to proceed without customer friction."
        )
    elif action == "review":
        if capped:
            what_decided = f"REVIEW (Safety Cap Downgrade) — Model score {risk_score:.4f} met BLOCK criteria, but hourly limit reached."
            why = (
                f"The XGBoost model recommended an immediate BLOCK due to severe risk factors, "
                f"but the hourly automated safety cap ({MAX_AUTO_BLOCKS_PER_HOUR} blocks/hour) is currently active. "
                f"To prevent systemic false-positive lockouts, VIGIL safely downgraded this payment to the priority human review queue."
            )
        else:
            what_decided = f"REVIEW (Step-Up Verification Required) — Risk score {risk_score:.4f} exceeds the {REVIEW_THRESHOLD} threshold."
            top_evidence = ", ".join(notices[:2]) if notices else "observed behavioral deviations"
            why = (
                f"Suspicious signals ({top_evidence}) increased the risk score. "
                f"However, the evidence does not cross the high-confidence automatic block cutoff ({BLOCK_THRESHOLD}). "
                f"VIGIL recommends step-up authentication (MFA) or manual review rather than automatically blocking the customer."
            )
    else:  # block
        what_decided = f"BLOCK (Declined / Held) — Risk score {risk_score:.4f} exceeds the {BLOCK_THRESHOLD} high-confidence cutoff."
        top_evidence = ", ".join(notices[:3]) if notices else "multiple anomalous factors"
        why = (
            f"Multiple high-risk signals were detected simultaneously ({top_evidence}). "
            f"The combination of severe behavioral deviation and contextual anomalies indicates probable account takeover or automated fraud. "
            f"VIGIL placed an immediate automated hold to protect merchant and cardholder funds."
        )

    return {
        "what_happened": what_happened,
        "what_noticed": notices_text,
        "what_decided": what_decided,
        "why": why,
    }


# ---------------------------------------------------------------------
# Defensive Table Renderer (PyArrow OS policy resilience)
# ---------------------------------------------------------------------
def render_df(df: pd.DataFrame, placeholder: Any = None, **kwargs):
    """
    Renders a DataFrame safely. Uses st.dataframe if supported by the runtime,
    or falls back gracefully to st.table if PyArrow is restricted by OS Application Control policies.
    """
    target = placeholder if placeholder is not None else st
    try:
        return target.dataframe(df, **kwargs)
    except Exception:
        return target.table(df)


# ---------------------------------------------------------------------
# Live Risk Simulation: Transaction Scenario Generators
# ---------------------------------------------------------------------
def generate_scenario_transactions(scenario_name: str, seed: int = 101) -> List[Dict[str, Any]]:
    """
    Generates realistic sequential transaction payloads for live simulation.
    All transactions are scored by the real pipeline with no duplicated logic.
    """
    rng = np.random.default_rng(seed)
    txns = []

    if scenario_name == "Normal Traffic":
        # 20 routine transactions from recognized users
        for _ in range(20):
            uid = int(rng.integers(10, 500))
            txns.append({
                "user_id": uid,
                "amount": round(float(rng.gamma(shape=2.0, scale=400)), 2),
                "time_since_last_txn_min": round(float(rng.exponential(scale=120)), 1),
                "txn_velocity_10min": int(rng.choice([0, 1], p=[0.85, 0.15])),
                "device_change": 0,
                "geo_dist_from_usual_km": round(float(rng.gamma(shape=1.2, scale=4)), 1),
                "login_burst_count": 0,
                "ip_risk_score": round(float(np.clip(rng.normal(0.08, 0.05), 0.01, 0.25)), 3),
                "hour_of_day": int(rng.integers(9, 21)),
            })

    elif scenario_name == "Suspicious Activity":
        # 25 transactions: 5 normal, then 20 with moderate anomalies
        for _ in range(5):
            uid = int(rng.integers(10, 500))
            txns.append({
                "user_id": uid,
                "amount": round(float(rng.gamma(shape=2.0, scale=400)), 2),
                "time_since_last_txn_min": round(float(rng.exponential(scale=90)), 1),
                "txn_velocity_10min": 0,
                "device_change": 0,
                "geo_dist_from_usual_km": round(float(rng.gamma(shape=1.2, scale=5)), 1),
                "login_burst_count": 0,
                "ip_risk_score": round(float(np.clip(rng.normal(0.10, 0.05), 0.02, 0.20)), 3),
                "hour_of_day": 14,
            })
        for i in range(20):
            uid = int(rng.integers(10, 500))
            susp_type = i % 4
            if susp_type == 0:
                txns.append({
                    "user_id": uid,
                    "amount": round(float(rng.uniform(14000, 22000)), 2),
                    "time_since_last_txn_min": round(float(rng.uniform(30, 90)), 1),
                    "txn_velocity_10min": 1,
                    "device_change": 0,
                    "geo_dist_from_usual_km": round(float(rng.uniform(2, 10)), 1),
                    "login_burst_count": 0,
                    "ip_risk_score": round(float(rng.uniform(0.20, 0.40)), 3),
                    "hour_of_day": 15,
                })
            elif susp_type == 1:
                txns.append({
                    "user_id": uid,
                    "amount": round(float(rng.uniform(1200, 3500)), 2),
                    "time_since_last_txn_min": round(float(rng.uniform(60, 180)), 1),
                    "txn_velocity_10min": 1,
                    "device_change": 1,
                    "geo_dist_from_usual_km": round(float(rng.uniform(15, 60)), 1),
                    "login_burst_count": 1,
                    "ip_risk_score": round(float(rng.uniform(0.35, 0.55)), 3),
                    "hour_of_day": 18,
                })
            elif susp_type == 2:
                txns.append({
                    "user_id": uid,
                    "amount": round(float(rng.uniform(800, 2500)), 2),
                    "time_since_last_txn_min": round(float(rng.uniform(2, 10)), 1),
                    "txn_velocity_10min": int(rng.integers(3, 5)),
                    "device_change": 0,
                    "geo_dist_from_usual_km": round(float(rng.uniform(1, 8)), 1),
                    "login_burst_count": 0,
                    "ip_risk_score": round(float(rng.uniform(0.15, 0.35)), 3),
                    "hour_of_day": 20,
                })
            else:
                txns.append({
                    "user_id": uid,
                    "amount": round(float(rng.uniform(2000, 6000)), 2),
                    "time_since_last_txn_min": round(float(rng.uniform(10, 45)), 1),
                    "txn_velocity_10min": 1,
                    "device_change": 0,
                    "geo_dist_from_usual_km": round(float(rng.uniform(10, 35)), 1),
                    "login_burst_count": 2,
                    "ip_risk_score": round(float(rng.uniform(0.30, 0.50)), 3),
                    "hour_of_day": 2,
                })

    elif scenario_name == "Account Takeover Burst":
        # 25 transactions: 5 normal baseline, then 20 severe ATO attacks
        for _ in range(5):
            uid = int(rng.integers(10, 500))
            txns.append({
                "user_id": uid,
                "amount": round(float(rng.gamma(shape=2.0, scale=400)), 2),
                "time_since_last_txn_min": 120.0,
                "txn_velocity_10min": 0,
                "device_change": 0,
                "geo_dist_from_usual_km": 2.5,
                "login_burst_count": 0,
                "ip_risk_score": 0.08,
                "hour_of_day": 14,
            })
        for _ in range(20):
            uid = int(rng.integers(500, 900))
            txns.append({
                "user_id": uid,
                "amount": round(float(rng.uniform(25000, 75000)), 2),
                "time_since_last_txn_min": round(float(rng.uniform(0.2, 2.0)), 1),
                "txn_velocity_10min": int(rng.integers(6, 12)),
                "device_change": 1,
                "geo_dist_from_usual_km": round(float(rng.uniform(800, 2400)), 1),
                "login_burst_count": int(rng.integers(4, 9)),
                "ip_risk_score": round(float(rng.uniform(0.82, 0.98)), 3),
                "hour_of_day": int(rng.choice([1, 2, 3, 4])),
            })

    elif scenario_name == "High-Risk Combined Attack":
        # 60 transactions: 5 normal, then 55 severe attacks to trip the 50/hour Safety Cap
        for _ in range(5):
            uid = int(rng.integers(10, 500))
            txns.append({
                "user_id": uid,
                "amount": round(float(rng.gamma(shape=2.0, scale=400)), 2),
                "time_since_last_txn_min": 120.0,
                "txn_velocity_10min": 0,
                "device_change": 0,
                "geo_dist_from_usual_km": 3.0,
                "login_burst_count": 0,
                "ip_risk_score": 0.05,
                "hour_of_day": 12,
            })
        for _ in range(55):
            uid = int(rng.integers(10, 950))
            txns.append({
                "user_id": uid,
                "amount": round(float(rng.uniform(30000, 85000)), 2),
                "time_since_last_txn_min": round(float(rng.uniform(0.1, 1.5)), 1),
                "txn_velocity_10min": int(rng.integers(7, 14)),
                "device_change": 1,
                "geo_dist_from_usual_km": round(float(rng.uniform(1000, 3000)), 1),
                "login_burst_count": int(rng.integers(5, 10)),
                "ip_risk_score": round(float(rng.uniform(0.85, 0.99)), 3),
                "hour_of_day": int(rng.choice([2, 3, 4])),
            })

    return txns


# ---------------------------------------------------------------------
# Load shared pipeline & evaluate baseline batch
# ---------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    pipeline = get_pipeline()
    df = generate_synthetic_data(n_txns=5000, seed=42)
    audit_df, audit_log = pipeline.score_batch(df, record_audit=False)
    pipeline.safety_cap.reset()
    return pipeline, audit_df, audit_log


with st.spinner("Initializing VIGIL risk pipeline, model artifacts, and user baselines..."):
    pipeline, audit_df, audit_log = load_pipeline()

# Dynamic system status metrics
model_ok = getattr(pipeline, "model", None) is not None
user_stats_dict = getattr(pipeline.profiler, "user_stats", {}) if getattr(pipeline, "profiler", None) is not None else {}
actual_user_count = len(user_stats_dict)
prof_ok = getattr(pipeline, "profiler", None) is not None and actual_user_count > 0
safe_ok = getattr(pipeline, "safety_cap", None) is not None
audit_ok = getattr(pipeline, "audit_logger", None) is not None


# ---------------------------------------------------------------------
# Sidebar Branding & Configuration
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## VIGIL")
    st.markdown("**Behavioral AI Risk Manager**")
    st.caption("Detect • Explain • Decide • Protect")
    st.divider()

    st.markdown("### Core Pillars")
    st.markdown(
        """
        - **Detect**: Behavioral $z$-score deviation
        - **Explain**: On-demand TreeSHAP attribution
        - **Decide**: Cost-weighted bounded actions
        - **Protect**: Thread-safe safety capping
        """
    )
    st.divider()

    st.markdown("### Engine Policy")
    st.markdown(
        f"""
        - **Decision Bounds**:
          - `ALLOW`: `< {REVIEW_THRESHOLD}`
          - `REVIEW`: `[{REVIEW_THRESHOLD}, {BLOCK_THRESHOLD})`
          - `BLOCK`: `≥ {BLOCK_THRESHOLD}`
        - **Safety Limit**: `{MAX_AUTO_BLOCKS_PER_HOUR} auto-blocks/hr`
        - **Model Core**: XGBoost (Cost-sensitive weighted trees)
        - **Explainability**: Local TreeSHAP
        - **Baseline Profiling**: Leak-Free `UserProfiler`
        """
    )
    st.divider()

    # Compact System Status Panel (Dynamically derived)
    st.markdown("### System Status")
    st.markdown(
        f"""
        <div style="background: #0d1424; border: 1px solid #1c273e; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.82rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: #64748b;">Risk Model</span>
                <span style="color: {'#10b981' if model_ok else '#ef4444'}; font-weight: 600;">{'Loaded' if model_ok else 'Offline'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: #64748b;">User Profiles</span>
                <span style="color: {'#10b981' if prof_ok else '#ef4444'}; font-weight: 600;">{'Loaded (' + f"{actual_user_count:,}" + ' users)' if prof_ok else 'Offline'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: #64748b;">Safety Controller</span>
                <span style="color: {'#10b981' if safe_ok else '#ef4444'}; font-weight: 600;">{'Active' if safe_ok else 'Offline'}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748b;">Audit Trail</span>
                <span style="color: {'#10b981' if audit_ok else '#ef4444'}; font-weight: 600;">{'Active' if audit_ok else 'Offline'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "VIGIL identifies high-velocity fraud spikes and account takeover (ATO) "
        "by profiling deviations from per-user historical spending baselines."
    )


# ---------------------------------------------------------------------
# Main Header & Landing Presentation
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class="vigil-banner">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.85rem; font-weight: 800; letter-spacing: -0.5px; color: #f8fafc;">
                    VIGIL
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #38bdf8; margin-top: 2px;">
                    Behavioral AI Risk Manager
                </div>
                <div style="font-size: 0.85rem; font-weight: 500; color: #94a3b8; margin-top: 4px; letter-spacing: 0.5px;">
                    Detect • Explain • Decide • Protect
                </div>
            </div>
            <div style="margin-top: 0.5rem;">
                <span style="background: rgba(30, 58, 138, 0.4); color: #93c5fd; border: 1px solid #1d4ed8; padding: 0.35rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;">
                    ENTERPRISE RISK OPERATIONS
                </span>
            </div>
        </div>
        <div class="vigil-status-strip">
            <div class="status-strip-item">
                <span class="status-strip-label">SYSTEM STATUS</span>
                <span class="status-strip-value status-val-active">Operational</span>
            </div>
            <div class="status-strip-item">
                <span class="status-strip-label">MODEL</span>
                <span class="status-strip-value">{'XGBoost (Cost-Weighted)' if model_ok else 'Offline'}</span>
            </div>
            <div class="status-strip-item">
                <span class="status-strip-label">SAFETY CAP</span>
                <span class="status-strip-value status-val-active">{'Active (50/hr limit)' if safe_ok else 'Offline'}</span>
            </div>
            <div class="status-strip-item">
                <span class="status-strip-label">USER PROFILES</span>
                <span class="status-strip-value status-val-active">{'Loaded (' + f"{actual_user_count:,}" + ' accounts)' if prof_ok else 'Offline'}</span>
            </div>
            <div class="status-strip-item">
                <span class="status-strip-label">DECISION RULES</span>
                <span class="status-strip-value">Allow &lt; {REVIEW_THRESHOLD} | Review [{REVIEW_THRESHOLD}, {BLOCK_THRESHOLD}) | Block &ge; {BLOCK_THRESHOLD}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 Pillars Value Proposition Cards
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.markdown(
        """
        <div class="pillar-card" style="border-top: 3px solid #3b82f6;">
            <div class="pillar-title">1. Detect</div>
            <div class="pillar-sub">Behavioral Anomaly</div>
            <div class="pillar-desc">Measures spend deviation (dynamic z-score) against historical user baselines.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p2:
    st.markdown(
        """
        <div class="pillar-card" style="border-top: 3px solid #8b5cf6;">
            <div class="pillar-title">2. Explain</div>
            <div class="pillar-sub">On-Demand TreeSHAP</div>
            <div class="pillar-desc">Generates instant local feature attributions on flagged alerts for analyst triage.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p3:
    st.markdown(
        """
        <div class="pillar-card" style="border-top: 3px solid #10b981;">
            <div class="pillar-title">3. Decide</div>
            <div class="pillar-sub">Cost-Weighted Action</div>
            <div class="pillar-desc">Routes transactions to ALLOW, Step-Up REVIEW (MFA), or BLOCK to minimize total loss.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p4:
    st.markdown(
        f"""
        <div class="pillar-card" style="border-top: 3px solid #ef4444;">
            <div class="pillar-title">4. Protect</div>
            <div class="pillar-sub">Safety Cap Engine</div>
            <div class="pillar-desc">Limits auto-blocks to {MAX_AUTO_BLOCKS_PER_HOUR}/hr to prevent systemic merchant lockouts.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    f"Operating Policy: ALLOW < {REVIEW_THRESHOLD} | REVIEW [{REVIEW_THRESHOLD}, {BLOCK_THRESHOLD}) | BLOCK ≥ {BLOCK_THRESHOLD} | "
    f"Safety Cap: {MAX_AUTO_BLOCKS_PER_HOUR} auto-blocks/hr"
)

tab_sim, tab_live, tab_feed, tab_audit = st.tabs([
    "What-If Fraud Simulator",
    "Live Risk Simulation",
    "Flagged Alert Feed",
    "Audit & Telemetry",
])


# =====================================================================
# TAB 1: What-If Fraud Risk Simulator
# =====================================================================
with tab_sim:
    st.markdown("### What-If Fraud Risk Simulator")
    st.caption(
        "Evaluate individual payments against the live VIGIL RiskPipeline. "
        "Explore how spend deviations, velocity bursts, unknown hardware, and location shifts interact with decision thresholds and TreeSHAP."
    )

    # Preset selection handler
    def apply_preset():
        selected = st.session_state.get("preset_selector")
        if selected in SCENARIO_PRESETS:
            p = SCENARIO_PRESETS[selected]
            st.session_state["sim_user_id"] = int(p["user_id"])
            st.session_state["sim_amount"] = float(p["amount"])
            st.session_state["sim_time_since"] = float(p["time_since_last_txn_min"])
            st.session_state["sim_velocity"] = int(p["txn_velocity_10min"])
            st.session_state["sim_device"] = int(p["device_change"])
            st.session_state["sim_geo_dist"] = float(p["geo_dist_from_usual_km"])
            st.session_state["sim_login_burst"] = int(p["login_burst_count"])
            st.session_state["sim_ip_risk"] = float(p["ip_risk_score"])
            st.session_state["sim_hour"] = int(p["hour_of_day"])

    # Initialize session state with Normal Transaction
    if "sim_amount" not in st.session_state:
        default_preset = SCENARIO_PRESETS["Normal Transaction"]
        st.session_state["preset_selector"] = "Normal Transaction"
        st.session_state["sim_user_id"] = int(default_preset["user_id"])
        st.session_state["sim_amount"] = float(default_preset["amount"])
        st.session_state["sim_time_since"] = float(default_preset["time_since_last_txn_min"])
        st.session_state["sim_velocity"] = int(default_preset["txn_velocity_10min"])
        st.session_state["sim_device"] = int(default_preset["device_change"])
        st.session_state["sim_geo_dist"] = float(default_preset["geo_dist_from_usual_km"])
        st.session_state["sim_login_burst"] = int(default_preset["login_burst_count"])
        st.session_state["sim_ip_risk"] = float(default_preset["ip_risk_score"])
        st.session_state["sim_hour"] = int(default_preset["hour_of_day"])

    # Scenario Selection Bar
    col_preset, col_desc = st.columns([1, 2])
    with col_preset:
        preset_names = list(SCENARIO_PRESETS.keys()) + ["Custom Input"]
        st.selectbox(
            "Select Scenario Preset",
            options=preset_names,
            key="preset_selector",
            on_change=apply_preset,
            help="Choose a pre-configured attack vector or explore custom inputs.",
        )

    with col_desc:
        current_preset = st.session_state.get("preset_selector")
        if current_preset in SCENARIO_PRESETS:
            preset_data = SCENARIO_PRESETS[current_preset]
            tags_html = " ".join([f"<span class='tag-pill'>{t}</span>" for t in preset_data.get("tags", [])])
            st.markdown(
                f"""
                <div class="scenario-box">
                    <div style="font-weight: 700; color: #38bdf8; font-size: 0.85rem; margin-bottom: 0.25rem;">
                        Scenario Context & Story
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
                        {preset_data['context']}
                    </div>
                    <div style="margin-top: 0.4rem;">
                        {tags_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="scenario-box">
                    <div style="font-weight: 700; color: #38bdf8; font-size: 0.85rem; margin-bottom: 0.25rem;">
                        Scenario Context & Story
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
                        Custom input mode. Adjust parameters freely to explore how deviations in spend, velocity, and network signals affect risk.
                    </div>
                    <div style="margin-top: 0.4rem;">
                        <span class="tag-pill">Custom Simulation</span>
                        <span class="tag-pill">Manual Exploration</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("##### Transaction Parameters")
    c1, c2 = st.columns(2)

    with c1:
        user_help_txt = (
            f"Known users (0–{actual_user_count - 1}) compare against their fitted historical baseline. "
            f"Unseen IDs ({actual_user_count}+) safely fall back to population prior."
            if actual_user_count > 0
            else "Unseen IDs safely fall back to population prior."
        )
        st.number_input(
            "User ID",
            min_value=0,
            max_value=10000,
            step=1,
            key="sim_user_id",
            help=user_help_txt,
        )
        st.number_input(
            "Transaction Amount (₹ INR)",
            min_value=1.0,
            max_value=1000000.0,
            step=500.0,
            key="sim_amount",
            help="Transaction amount in INR. Dynamically compared against user baseline to derive amount z-score (deviation).",
        )
        st.number_input(
            "Time Since Previous Transaction (minutes)",
            min_value=0.0,
            max_value=10000.0,
            step=5.0,
            key="sim_time_since",
            help="Elapsed minutes since the cardholder's prior transaction.",
        )
        st.number_input(
            "Transaction Velocity (past 10 min)",
            min_value=0,
            max_value=50,
            step=1,
            key="sim_velocity",
            help="Count of transactions initiated in the preceding 10-minute window.",
        )
        st.selectbox(
            "Device Status",
            options=[0, 1],
            format_func=lambda x: "0 — Recognized / Familiar Device" if x == 0 else "1 — New Hardware / Unknown Fingerprint",
            key="sim_device",
            help="Flags whether the incoming request originated from a familiar hardware device or newly seen fingerprint.",
        )

    with c2:
        st.number_input(
            "Distance from Usual Location (km)",
            min_value=0.0,
            max_value=5000.0,
            step=10.0,
            key="sim_geo_dist",
            help="Distance in kilometers from user's historical centroid / habitual transaction locus.",
        )
        st.number_input(
            "Recent Login Burst Count (past 5 min)",
            min_value=0,
            max_value=20,
            step=1,
            key="sim_login_burst",
            help="Count of login events and authentication attempts in the 5 minutes preceding the payment.",
        )
        st.slider(
            "Network / IP Risk Score",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="sim_ip_risk",
            help="External threat intelligence score: 0.0 (clean domestic residential) to 1.0 (TOR exit node / malicious proxy).",
        )
        st.slider(
            "Hour of Day (0 – 23)",
            min_value=0,
            max_value=23,
            step=1,
            key="sim_hour",
            help="Hour of transaction origination in local time (e.g., 2 = 2 AM, 14 = 2 PM).",
        )

    analyze_clicked = st.button("Analyze Transaction", type="primary", use_container_width=True)

    if analyze_clicked:
        txn_payload = {
            "user_id": int(st.session_state["sim_user_id"]),
            "amount": float(st.session_state["sim_amount"]),
            "time_since_last_txn_min": float(st.session_state["sim_time_since"]),
            "txn_velocity_10min": int(st.session_state["sim_velocity"]),
            "device_change": int(st.session_state["sim_device"]),
            "geo_dist_from_usual_km": float(st.session_state["sim_geo_dist"]),
            "login_burst_count": int(st.session_state["sim_login_burst"]),
            "ip_risk_score": float(st.session_state["sim_ip_risk"]),
            "hour_of_day": int(st.session_state["sim_hour"]),
        }

        # Calculate baseline z-score deviation for UI transparency
        zscore = pipeline.profiler.transform_single(txn_payload)

        # Execute scoring via the production-aligned RiskPipeline
        sim_result = pipeline.score_single(txn_payload, record_audit=True)
        st.session_state["last_sim_result"] = (txn_payload, sim_result, zscore)

    # -----------------------------------------------------------------
    # Render Simulation Results (Visual Focal Point)
    # -----------------------------------------------------------------
    if "last_sim_result" in st.session_state:
        txn_payload, result, zscore = st.session_state["last_sim_result"]
        risk_score = result["risk_score"]
        action = result["action"]
        capped = result["capped_by_safety_limit"]
        txn_id = result["transaction_id"]

        # Generate humanized storytelling
        story = generate_human_explanation(txn_payload, result, zscore)

        st.divider()
        st.markdown("### Risk Evaluation Results")

        # 1. Visually Dominant Result Card (Requirement 4)
        cap_badge_html = "<span class='badge-capped'>SAFETY CAP ACTIVE</span>" if capped else ""
        st.markdown(
            f"""
            <div class="result-card result-{action}">
                <div class="result-top-row">
                    <div>
                        <div class="result-kicker">RISK SCORE</div>
                        <div class="result-score-num">{risk_score:.4f}</div>
                    </div>
                    <div>
                        <span class="badge-{action}">{action.upper()}</span>
                        {cap_badge_html}
                    </div>
                </div>
                <div class="result-rationale-box">
                    <div class="result-rationale-label">DECISION RATIONALE</div>
                    <div class="result-rationale-desc">{story['why']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Horizontal Risk Score Meter (Requirement 5)
        marker_pct = min(max(risk_score, 0.0), 1.0) * 100.0
        zone_allow_w = REVIEW_THRESHOLD * 100.0
        zone_review_w = (BLOCK_THRESHOLD - REVIEW_THRESHOLD) * 100.0
        zone_block_w = (1.0 - BLOCK_THRESHOLD) * 100.0

        st.markdown(
            f"""
            <div class="risk-meter-wrapper">
                <div class="risk-meter-header">
                    <span class="risk-meter-title">RISK SCORE METER</span>
                    <span class="risk-meter-current">Score: <strong>{risk_score:.4f}</strong></span>
                </div>
                <div class="risk-meter-track-container">
                    <div class="risk-meter-track">
                        <div class="track-zone zone-allow" style="width: {zone_allow_w:.1f}%;">
                            ALLOW
                        </div>
                        <div class="track-zone zone-review" style="width: {zone_review_w:.1f}%;">
                            REVIEW (Step-Up MFA)
                        </div>
                        <div class="track-zone zone-block" style="width: {zone_block_w:.1f}%;">
                            BLOCK
                        </div>
                    </div>
                    <div class="risk-meter-marker" style="left: {marker_pct:.2f}%;">
                        <div class="marker-line"></div>
                        <div class="marker-pill">{risk_score:.4f}</div>
                    </div>
                </div>
                <div class="risk-meter-ticks">
                    <span>0.00</span>
                    <span style="position: absolute; left: {zone_allow_w:.1f}%; transform: translateX(-50%);">{REVIEW_THRESHOLD:.2f}</span>
                    <span style="position: absolute; left: {BLOCK_THRESHOLD * 100:.1f}%; transform: translateX(-50%);">{BLOCK_THRESHOLD:.2f}</span>
                    <span>1.00</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. High-Density Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Risk Score", f"{risk_score:.4f}")
        action_labels = {
            "allow": "ALLOW",
            "review": "REVIEW (MFA)",
            "block": "BLOCK (Hold)",
        }
        m2.metric("Decision Action", action_labels.get(action, action.upper()))
        z_label = "Anomaly" if abs(zscore) >= 3.0 else "Baseline"
        m3.metric(f"Spend Deviation ({z_label})", f"{zscore:+.2f}σ")
        cap_status = "Capped (Max 50)" if capped else "Normal (< 50/hr)"
        m4.metric("Safety Cap Status", cap_status)

        # 4. Executive Decision Rationale (Requirement 4)
        st.markdown("##### Executive Decision Rationale")
        q1, q2 = st.columns(2)
        with q1:
            st.markdown(
                f"""
                <div class="scenario-box">
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">
                        1. What Happened?
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.86rem; margin-top: 0.3rem; line-height: 1.45;">
                        {story['what_happened']}
                    </div>
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; margin-top: 0.75rem;">
                        2. What Did VIGIL Notice?
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.84rem; margin-top: 0.3rem; line-height: 1.45;">
                        {story['what_noticed']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q2:
            st.markdown(
                f"""
                <div class="scenario-box">
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">
                        3. What Did VIGIL Decide?
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.86rem; margin-top: 0.3rem; line-height: 1.45;">
                        {story['what_decided']}
                    </div>
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; margin-top: 0.75rem;">
                        4. Why?
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.84rem; margin-top: 0.3rem; line-height: 1.45;">
                        {story['why']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 5. Explainability & Feature Attribution (Requirement 6)
        if action in ("review", "block"):
            st.markdown("### WHY WAS THIS TRANSACTION FLAGGED?")
            st.markdown(
                f"""
                <div class="scenario-box" style="border-left: 4px solid {'#ef4444' if action == 'block' else '#f59e0b'};">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; flex-wrap: wrap; font-size: 0.88rem;">
                        <span><strong>Risk Score:</strong> <code>{risk_score:.4f}</code></span>
                        <span><strong>Decision:</strong> <strong style="color: {'#ef4444' if action == 'block' else '#f59e0b'};">{action.upper()}</strong></span>
                        <span><strong>Safety Cap:</strong> {'Downgraded' if capped else 'Normal'}</span>
                    </div>
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; margin-top: 0.35rem; margin-bottom: 0.25rem;">
                        Executive Risk Rationale:
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.45;">
                        {story['why']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result.get("top_factors"):
                if result.get("explanation_degraded"):
                    st.warning("SHAP explanation service temporarily unavailable. Transaction was safely scored via fail-safe path.")
                else:
                    st.caption("Top Risk Contributors (TreeSHAP log-odds attribution):")
                    factors_df = pd.DataFrame(result["top_factors"]).rename(columns={
                        "feature": "Feature",
                        "value": "Observed Value",
                        "contribution": "Contribution",
                        "direction": "Direction",
                    })
                    factors_df["Direction"] = factors_df["Direction"].apply(
                        lambda d: "Increases Risk" if d == "increased" else "Decreases Risk"
                    )
                    render_df(factors_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No individual SHAP factors exceeded the significance threshold.")
        else:
            st.info(
                "Compute Optimization: Detailed TreeSHAP attribution is bypassed for approved transactions (< 0.05) "
                "to preserve real-time authorization throughput."
            )

        st.caption(f"Immutable Audit Record Logged: `{txn_id}` at `{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}`.")


# =====================================================================
# TAB 2: Live Risk Simulation
# =====================================================================
with tab_live:
    st.markdown("### LIVE RISK SIMULATION")
    st.caption(
        "Real-time risk operations console. Stream transaction flows to observe "
        "behavioral deviation detection, risk scoring spikes, and safety cap controls."
    )

    SIM_SCENARIO_INFO = {
        "Normal Traffic": {
            "count": 20,
            "description": "Simulates 20 routine customer payments from recognized users and locations with normal spend amounts.",
            "expected": "All transactions approved (ALLOW < 0.05). Risk score remains near zero.",
        },
        "Suspicious Activity": {
            "count": 25,
            "description": "5 baseline transactions followed by 20 payments with moderate deviations (unusual amounts, new devices, or elevated velocity).",
            "expected": "Triggers REVIEW [0.05, 0.80) routing payments to step-up authentication (2FA).",
        },
        "Account Takeover Burst": {
            "count": 25,
            "description": "5 normal transactions followed by a rapid attack burst of 20 compromised transactions (severe geo jumps, login bursts, new devices).",
            "expected": "Sharp risk score spike to 0.90+ triggering automatic BLOCK decisions.",
        },
        "High-Risk Combined Attack": {
            "count": 60,
            "description": "Relentless 60-transaction distributed attack designed to stress-test the 50/hour circuit breaker.",
            "expected": "First 50 blocks succeed; excess attacks hit the hourly safety cap and are downgraded to REVIEW.",
        },
    }

    sim_col1, sim_col2 = st.columns([3, 2])
    with sim_col1:
        scenario_choice = st.selectbox(
            "Select Risk Scenario",
            list(SIM_SCENARIO_INFO.keys()),
            index=2,
            key="live_sim_scenario_select",
        )
    with sim_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            start_clicked = st.button("Start Simulation", type="primary", use_container_width=True)
        with btn_c2:
            reset_clicked = st.button("Reset Simulation", use_container_width=True)
        with btn_c3:
            reset_cap_clicked = st.button("Reset Safety Cap", use_container_width=True)

    info = SIM_SCENARIO_INFO[scenario_choice]
    st.markdown(
        f"""
        <div class="scenario-box" style="margin-top: 0.5rem; margin-bottom: 1.25rem;">
            <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 600;">
                <strong>{scenario_choice}</strong> ({info['count']} transactions)
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.25rem;">
                {info['description']}
            </div>
            <div style="font-size: 0.8rem; color: #38bdf8; margin-top: 0.35rem;">
                <strong>Expected Behavior:</strong> {info['expected']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reset_cap_clicked:
        pipeline.safety_cap.reset()
        st.success("Hourly safety cap reset to 0 blocks.")

    if reset_clicked:
        if "live_sim_results" in st.session_state:
            del st.session_state["live_sim_results"]
        if "live_sim_scenario" in st.session_state:
            del st.session_state["live_sim_scenario"]
        st.rerun()

    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    feed_placeholder = st.empty()
    incident_placeholder = st.empty()

    if start_clicked:
        txns_to_run = generate_scenario_transactions(scenario_choice)
        live_results = []
        progress_bar = st.progress(0.0)

        for i, txn in enumerate(txns_to_run):
            res = pipeline.score_single(txn, record_audit=True)
            res["seq_num"] = i + 1
            res["amount_raw"] = txn["amount"]
            res["txn_input"] = txn
            live_results.append(res)

            total_n = len(live_results)
            allows = sum(1 for r in live_results if r["action"] == "allow")
            reviews = sum(1 for r in live_results if r["action"] == "review")
            blocks = sum(1 for r in live_results if r["action"] == "block")
            downgraded = sum(1 for r in live_results if r.get("capped_by_safety_limit"))

            with metrics_placeholder.container():
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("TRANSACTIONS", total_n)
                m2.metric("ALLOWED", f"{allows} ({allows/total_n:.0%})")
                m3.metric("REVIEW", f"{reviews} ({reviews/total_n:.0%})")
                m4.metric("BLOCKED", f"{blocks} ({blocks/total_n:.0%})")
                m5.metric("SAFETY DOWNGRADES", downgraded)

            scores = [r["risk_score"] for r in live_results]
            chart_df = pd.DataFrame({"Risk Score": scores}, index=range(1, len(scores) + 1))
            chart_placeholder.line_chart(chart_df, height=220)

            display_rows = []
            for r in live_results[-8:]:
                status_str = "Safety Cap: BLOCK downgraded to REVIEW" if r.get("capped_by_safety_limit") else "Normal"
                display_rows.append({
                    "Txn ID": f"TXN-{r['seq_num']:03d}",
                    "Amount": f"₹{r['amount_raw']:,.2f}",
                    "Risk Score": f"{r['risk_score']:.4f}",
                    "Decision": r["action"].upper(),
                    "Safety Cap Status": status_str,
                })
            render_df(pd.DataFrame(display_rows), placeholder=feed_placeholder, hide_index=True, use_container_width=True)

            progress_bar.progress((i + 1) / len(txns_to_run))
            time.sleep(0.06)

        progress_bar.empty()
        st.session_state["live_sim_results"] = live_results
        st.session_state["live_sim_scenario"] = scenario_choice

    # Display persisted simulation results
    if "live_sim_results" in st.session_state:
        res_list = st.session_state["live_sim_results"]
        scen = st.session_state.get("live_sim_scenario", "Simulation")
        total_n = len(res_list)
        allows = sum(1 for r in res_list if r["action"] == "allow")
        reviews = sum(1 for r in res_list if r["action"] == "review")
        blocks = sum(1 for r in res_list if r["action"] == "block")
        high_risk_n = sum(1 for r in res_list if r["risk_score"] >= REVIEW_THRESHOLD)
        downgraded = sum(1 for r in res_list if r.get("capped_by_safety_limit"))

        with metrics_placeholder.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("TRANSACTIONS", total_n)
            m2.metric("ALLOWED", f"{allows} ({allows/total_n:.0%})")
            m3.metric("REVIEW", f"{reviews} ({reviews/total_n:.0%})")
            m4.metric("BLOCKED", f"{blocks} ({blocks/total_n:.0%})")
            m5.metric("SAFETY DOWNGRADES", downgraded)

        scores = [r["risk_score"] for r in res_list]
        chart_df = pd.DataFrame({"Risk Score": scores}, index=range(1, len(scores) + 1))
        chart_placeholder.line_chart(chart_df, height=220)

        st.markdown("##### Sequential Transaction Log")
        full_display = []
        for r in res_list:
            status_str = "Safety Cap: BLOCK downgraded to REVIEW" if r.get("capped_by_safety_limit") else "Normal"
            full_display.append({
                "Txn ID": f"TXN-{r['seq_num']:03d}",
                "Amount": f"₹{r['amount_raw']:,.2f}",
                "Risk Score": f"{r['risk_score']:.4f}",
                "Decision": r["action"].upper(),
                "Safety Cap Status": status_str,
            })
        render_df(pd.DataFrame(full_display), placeholder=feed_placeholder, hide_index=True, use_container_width=True)

        # Incident Summary (Requirement 9)
        flagged_txns = [r["txn_input"] for r in res_list if r["risk_score"] >= REVIEW_THRESHOLD]
        signals = []
        if flagged_txns:
            max_geo = max(t.get("geo_dist_from_usual_km", 0) for t in flagged_txns)
            if max_geo > 50:
                signals.append(f"• **Geographic Deviation**: Locations up to {max_geo:.1f} km away from historical baseline")
            new_dev_count = sum(1 for t in flagged_txns if t.get("device_change") == 1)
            if new_dev_count > 0:
                signals.append(f"• **Unrecognized Hardware**: {new_dev_count} transactions originated from new/unverified devices")
            max_vel = max(t.get("txn_velocity_10min", 0) for t in flagged_txns)
            if max_vel >= 3:
                signals.append(f"• **Velocity Burst**: Transaction velocity peaked at {max_vel} transactions within 10 minutes")
            max_log = max(t.get("login_burst_count", 0) for t in flagged_txns)
            if max_log >= 2:
                signals.append(f"• **Authentication Burst**: Up to {max_log} rapid login attempts prior to transaction")
            max_ip = max(t.get("ip_risk_score", 0) for t in flagged_txns)
            if max_ip >= 0.40:
                signals.append(f"• **Threat Network / IP Risk**: Elevated IP threat score reaching {max_ip:.2f} (proxy/datacenter signature)")
            max_amt = max(t.get("amount", 0) for t in flagged_txns)
            if max_amt > 10000:
                signals.append(f"• **Amount Deviation**: High-value transactions up to ₹{max_amt:,.2f} exceeding spending baselines")
        else:
            signals.append("• **Nominal Traffic**: All behavioral features remained within acceptable per-user baseline boundaries.")

        signals_html = "<br>".join(signals)
        downgrade_html = ""
        if downgraded > 0:
            downgrade_html = f"""
            <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid #8b5cf6; border-radius: 4px; padding: 8px 12px; margin-top: 10px; color: #ddd6fe; font-size: 0.82rem;">
                <strong>Safety Valve Engaged:</strong> {downgraded} transactions exceeded the {MAX_AUTO_BLOCKS_PER_HOUR}/hour auto-block ceiling and were safely downgraded to <strong>REVIEW</strong> (Step-Up Auth) to protect business continuity and legitimate merchants.
            </div>
            """

        incident_placeholder.markdown(
            f"""
            <div class="scenario-box" style="border-left: 4px solid #ef4444; margin-top: 1.25rem;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #f87171; margin-bottom: 0.5rem;">
                    INCIDENT SUMMARY — {scen.upper()}
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 0.75rem; font-size: 0.82rem;">
                    <div><span style="color: #64748b;">Scenario:</span> <strong>{scen}</strong></div>
                    <div><span style="color: #64748b;">Transactions Analyzed:</span> <strong>{total_n}</strong></div>
                    <div><span style="color: #64748b;">High-Risk Events:</span> <strong style="color: #fbbf24;">{high_risk_n}</strong></div>
                    <div><span style="color: #64748b;">Automatic Blocks:</span> <strong style="color: #f87171;">{blocks}</strong></div>
                    <div><span style="color: #64748b;">Step-Up Reviews:</span> <strong style="color: #fde68a;">{reviews}</strong></div>
                    <div><span style="color: #64748b;">Safety Downgrades:</span> <strong style="color: #a78bfa;">{downgraded}</strong></div>
                </div>
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.78rem; text-transform: uppercase; margin-top: 0.5rem; margin-bottom: 0.35rem;">
                    PRIMARY SUSPICIOUS SIGNALS:
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">
                    {signals_html}
                </div>
                {downgrade_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # On-Demand TreeSHAP inspection
        flagged_entries = [r for r in res_list if r["action"] in ("review", "block")]
        if flagged_entries:
            st.divider()
            st.markdown("##### Flagged Transaction Detail (TreeSHAP)")
            flagged_options = {
                f"TXN-{r['seq_num']:03d} (₹{r['amount_raw']:,.2f} | Score {r['risk_score']:.4f} | {r['action'].upper()})": r
                for r in flagged_entries
            }
            sel_label = st.selectbox("Select flagged transaction to view explainability attribution:", list(flagged_options.keys()), key="live_flagged_select")
            selected_entry = flagged_options[sel_label]

            st.markdown(
                f"""
                <div class="scenario-box" style="border-left: 4px solid {'#ef4444' if selected_entry['action'] == 'block' else '#f59e0b'};">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">
                        WHY WAS THIS TRANSACTION FLAGGED?
                    </div>
                    <div style="font-size: 0.86rem; margin-bottom: 0.2rem;">
                        <strong>Risk Score:</strong> <code>{selected_entry['risk_score']:.4f}</code> &nbsp;|&nbsp;
                        <strong>Decision:</strong> <span style="color: {'#ef4444' if selected_entry['action'] == 'block' else '#f59e0b'}; font-weight: 700;">{selected_entry['action'].upper()}</span> &nbsp;|&nbsp;
                        <strong>Safety Cap:</strong> {'Downgraded' if selected_entry.get('capped_by_safety_limit') else 'Normal'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if selected_entry.get("top_factors"):
                f_df = pd.DataFrame(selected_entry["top_factors"]).rename(columns={
                    "feature": "Behavioral Signal",
                    "value": "Observed Value",
                    "contribution": "SHAP Contribution",
                    "direction": "Risk Direction",
                })
                f_df["Risk Direction"] = f_df["Risk Direction"].apply(
                    lambda d: "Increases Risk" if d == "increased" else "Decreases Risk"
                )
                render_df(f_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No individual SHAP factors exceeded significance threshold.")


# =====================================================================
# TAB 3: Flagged Alert Feed (Batch Monitoring)
# =====================================================================
with tab_feed:
    st.markdown("### FLAGGED ALERT FEED")
    st.caption(
        f"Evaluated on {len(audit_df):,} synthetic transactions | "
        f"Decision Rules: Review ≥ {REVIEW_THRESHOLD} | Block ≥ {BLOCK_THRESHOLD} | Safety Cap: {MAX_AUTO_BLOCKS_PER_HOUR}/hr"
    )

    flagged = audit_df[audit_df["action"].isin(["review", "block"])]
    precision_on_flagged = flagged["actual_label"].mean() if len(flagged) else 0
    missed_fraud = len(audit_df[(audit_df["action"] == "allow") & (audit_df["actual_label"] == 1)])
    capped_count = audit_df["capped_by_safety_limit"].sum()

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Precision on Flagged", f"{precision_on_flagged:.1%}")
    b2.metric("Fraud Missed (FN)", missed_fraud)
    b3.metric("Auto-Blocked", len(audit_df[audit_df["action"] == "block"]))
    b4.metric("Downgraded by Safety Cap", int(capped_count))

    st.divider()
    st.markdown("##### Flagged Queue (Select a row to inspect SHAP explanation)")

    # Derive Primary Reason and Time for each flagged row (Requirement 10)
    def derive_primary_reason(idx):
        entry = audit_log[idx]
        factors = entry.get("top_factors")
        if factors and len(factors) > 0:
            feat = factors[0].get("feature", "Behavioral Anomaly")
            feat_labels = {
                "amount_zscore": "Spend Deviation (z-score)",
                "geo_dist_from_usual_km": "Geographic Distance",
                "txn_velocity_10min": "Transaction Velocity",
                "device_change": "Unrecognized Device",
                "login_burst_count": "Login Burst Count",
                "ip_risk_score": "Network / IP Threat",
                "time_since_last_txn_min": "Time Since Last Txn",
                "hour_of_day": "Off-Hours Transaction",
            }
            return feat_labels.get(feat, feat)
        return "Baseline Spending"

    feed_rows = []
    for orig_idx in flagged.index:
        entry = audit_log[orig_idx]
        t_str = entry.get("timestamp", "")
        try:
            time_formatted = datetime.fromisoformat(t_str).strftime("%H:%M:%S UTC")
        except Exception:
            time_formatted = t_str[-12:-4] if len(t_str) >= 12 else "N/A"

        feed_rows.append({
            "Transaction ID": entry["transaction_id"],
            "Risk Score": f"{entry['risk_score']:.4f}",
            "Decision": entry["action"].upper(),
            "Primary Reason": derive_primary_reason(orig_idx),
            "Time": time_formatted,
            "Capped": "Yes" if entry.get("capped_by_safety_limit") else "No",
            "_orig_idx": orig_idx,
        })

    feed_df = pd.DataFrame(feed_rows).sort_values("Risk Score", ascending=False).reset_index(drop=True)
    display_feed_df = feed_df[["Transaction ID", "Risk Score", "Decision", "Primary Reason", "Time", "Capped"]]

    has_selection = False
    selected_txn_id = None
    try:
        selected_idx = st.dataframe(
            display_feed_df,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
        )
        if selected_idx and selected_idx.get("selection", {}).get("rows"):
            row_num = selected_idx["selection"]["rows"][0]
            selected_txn_id = display_feed_df.iloc[row_num]["Transaction ID"]
            has_selection = True
    except Exception:
        render_df(display_feed_df, hide_index=True, use_container_width=True)
        flagged_ids = display_feed_df["Transaction ID"].tolist()
        if flagged_ids:
            selected_txn_id = st.selectbox("Select flagged transaction to inspect:", flagged_ids, key="batch_flagged_select")
            has_selection = bool(selected_txn_id)

    st.divider()
    st.markdown("##### Flagged Alert Detail")
    if has_selection and selected_txn_id:
        match_rows = [i for i, e in enumerate(audit_log) if e["transaction_id"] == selected_txn_id]
        if match_rows:
            entry = audit_log[match_rows[0]]
            st.markdown(
                f"**Transaction**: `{entry['transaction_id']}` | "
                f"**Risk Score**: `{entry['risk_score']}` | "
                f"**Action**: `{entry['action'].upper()}`"
            )

            if entry["top_factors"]:
                factor_df = pd.DataFrame(entry["top_factors"]).rename(columns={
                    "feature": "Feature",
                    "value": "Value",
                    "contribution": "Contribution",
                    "direction": "Direction",
                })
                factor_df["Direction"] = factor_df["Direction"].apply(
                    lambda d: "Increases Risk" if d == "increased" else "Decreases Risk"
                )
                render_df(factor_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No explanation generated for this entry.")

            if entry.get("capped_by_safety_limit"):
                st.warning(
                    f"This block was downgraded to review — the {MAX_AUTO_BLOCKS_PER_HOUR}/hour "
                    "safety cap had already been reached."
                )
    else:
        st.info("Click any row in the flagged queue above to inspect its real-time TreeSHAP explanation.")


# =====================================================================
# TAB 4: Audit Trail & System Telemetry
# =====================================================================
with tab_audit:
    st.markdown("### AUDIT & TELEMETRY")
    st.caption("Real-time telemetry counters, safety cap sliding-window state, and append-only audit trail.")

    t_stats = pipeline.telemetry.get_stats()
    current_blocks = pipeline.safety_cap.current_block_count
    max_blocks = pipeline.safety_cap.max_blocks_per_hour
    remaining_cap = max(0, max_blocks - current_blocks)

    # 1. SYSTEM TELEMETRY (Requirement 11)
    st.markdown("##### SYSTEM TELEMETRY")
    act1, act2, act3, act4 = st.columns(4)
    act1.metric("Requests", t_stats["total_requests"])
    act2.metric("Allows", t_stats["allow_count"])
    act3.metric("Reviews", t_stats["review_count"])
    act4.metric("Blocks", t_stats["block_count"])

    # 2. SAFETY CONTROLLER (Requirement 11)
    st.markdown("##### SAFETY CONTROLLER")
    saf1, saf2, saf3 = st.columns(3)
    saf1.metric("Current Block Count", f"{current_blocks} / {max_blocks}")
    saf2.metric("Configured Limit", f"{max_blocks} / hr")
    saf3.metric("Window Status", "Active (Downgrading)" if current_blocks >= max_blocks else "Within Limit (Available)")

    # 3. RELIABILITY & ARTIFACTS
    st.markdown("##### RELIABILITY & ARTIFACTS")
    h1, h2, h3 = st.columns([2, 2, 1])
    with h1:
        st.caption(
            f"Reliability metrics: Degraded SHAP explanations: `{t_stats['degraded_explanation_count']}` | "
            f"Audit log I/O failures: `{t_stats['audit_failure_count']}`"
        )
    with h2:
        st.caption(
            f"Active Artifacts: `artifacts/model.json` (Loaded) | `artifacts/user_baselines.json` ({actual_user_count:,} accounts)"
        )
    with h3:
        if st.button("Reset Safety Cap", use_container_width=True, key="audit_reset_cap"):
            pipeline.safety_cap.reset()
            st.success("Safety cap counter reset.")
            st.rerun()

    st.divider()
    # 4. IMMUTABLE AUDIT TRAIL (Requirement 11)
    st.markdown("##### AUDIT TRAIL")
    st.caption("Showing most recent 100 entries from persistent `audit_trail.jsonl`.")

    live_entries = []
    if os.path.exists("audit_trail.jsonl"):
        try:
            with open("audit_trail.jsonl", "r", encoding="utf-8") as f:
                live_entries = [json.loads(line) for line in f if line.strip()]
        except Exception:
            live_entries = audit_log
    else:
        live_entries = audit_log

    if live_entries:
        live_audit_df = pd.DataFrame(live_entries)
        display_cols = [
            c for c in [
                "timestamp", "transaction_id", "risk_score", "action",
                "capped_by_safety_limit", "actual_label"
            ] if c in live_audit_df.columns
        ]
        audit_display = live_audit_df.tail(100)[display_cols].reset_index(drop=True)
        render_df(audit_display, hide_index=True, use_container_width=True)

        st.download_button(
            "Download Full Audit Trail (JSONL)",
            data="\n".join(json.dumps(e) for e in live_entries),
            file_name="audit_trail.jsonl",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Audit log is currently empty.")
