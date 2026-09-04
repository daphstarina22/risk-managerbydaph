"""
Fraud-spike / account-takeover detector — Streamlit dashboard & What-If Simulator.
Razorpay Buildathon — AI Risk Manager track.

Features:
  1. What-If Fraud Risk Simulator: Interactive transaction feature tweaking with 6 presets,
     dynamic baseline z-score derivation, bounded actions, safety cap, and on-demand SHAP.
  2. Flagged Alert Feed: Batch transaction feed with precision/recall metrics and per-alert SHAP.
  3. Audit Trail & System Telemetry: Live append-only audit trail and in-process operational metrics.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd
import streamlit as st

from decision_layer import (
    BLOCK_THRESHOLD,
    MAX_AUTO_BLOCKS_PER_HOUR,
    REVIEW_THRESHOLD,
)
from fraud_classifier import generate_synthetic_data
from risk_pipeline import get_pipeline


st.set_page_config(
    page_title="Risk Manager — AI Fraud & ATO Monitor",
    page_icon="🛡️",
    layout="wide",
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
        "description": "Typical daytime transaction (₹450) from known device and usual location with low IP risk.",
    },
    "Unusual Amount": {
        "user_id": 0,
        "amount": 18500.0,
        "time_since_last_txn_min": 45.0,
        "txn_velocity_10min": 1,
        "device_change": 0,
        "geo_dist_from_usual_km": 3.0,
        "login_burst_count": 0,
        "ip_risk_score": 0.10,
        "hour_of_day": 16,
        "description": "High ticket amount (₹18,500) causing extreme deviation against historical user baseline, but from familiar device and location.",
    },
    "New Device": {
        "user_id": 105,
        "amount": 750.0,
        "time_since_last_txn_min": 90.0,
        "txn_velocity_10min": 1,
        "device_change": 1,
        "geo_dist_from_usual_km": 8.0,
        "login_burst_count": 1,
        "ip_risk_score": 0.35,
        "hour_of_day": 19,
        "description": "Low-to-moderate purchase from a new hardware fingerprint with elevated IP risk.",
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
        "description": "Burst of 5 transactions within 10 minutes and 0.8 min since last payment, indicating rapid automated card testing.",
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
        "description": "Credential stuffing attack at 2 AM: multiple rapid failed logins, sudden 320 km location jump, new device, and instant ₹12,000 transfer.",
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
        "description": "Severe multi-vector attack: proxy/tor IP risk (0.98), 5 burst logins, extreme velocity, 520 km jump, and ₹35,000 extraction at 3 AM.",
    },
}


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


with st.spinner("Initializing shared risk pipeline, model artifacts, and baselines..."):
    pipeline, audit_df, audit_log = load_pipeline()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🛡️ AI Risk Manager — Fraud-Spike & ATO Monitor")
st.caption(
    "Razorpay AI Buildathon — AI Risk Manager Track | "
    f"Operating Thresholds: Allow < {REVIEW_THRESHOLD} | Review [{REVIEW_THRESHOLD}, {BLOCK_THRESHOLD}) | Block ≥ {BLOCK_THRESHOLD} | "
    f"Safety Cap: {MAX_AUTO_BLOCKS_PER_HOUR} auto-blocks/hr"
)

tab_sim, tab_feed, tab_audit = st.tabs([
    "⚡ What-If Fraud Simulator",
    "📊 Flagged Alert Feed (Batch)",
    "📜 Audit Trail & Telemetry",
])


# =====================================================================
# TAB 1: What-If Fraud Risk Simulator
# =====================================================================
with tab_sim:
    st.subheader("Interactive What-If Fraud Risk Simulator")
    st.markdown(
        "Evaluate individual transactions against the live `RiskPipeline`. "
        "Test how sudden spending spikes, device changes, geographic velocity, "
        "and login bursts interact with the machine learning model, decision boundaries, "
        "safety cap, and on-demand SHAP explanations."
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

    col_preset, col_desc = st.columns([1, 2])
    with col_preset:
        preset_names = list(SCENARIO_PRESETS.keys()) + ["Custom Input"]
        st.selectbox(
            "Select Scenario Preset",
            options=preset_names,
            key="preset_selector",
            on_change=apply_preset,
        )

    with col_desc:
        current_preset = st.session_state.get("preset_selector")
        if current_preset in SCENARIO_PRESETS:
            st.info(f"**Pattern**: {SCENARIO_PRESETS[current_preset]['description']}")
        else:
            st.info("**Pattern**: Custom scenario. Adjust sliders and parameters freely below.")

    st.markdown("##### ⚙️ Transaction Parameters")
    c1, c2 = st.columns(2)

    with c1:
        st.number_input(
            "User ID (0 – 499 for historical profile baselines)",
            min_value=0,
            max_value=1000,
            step=1,
            key="sim_user_id",
            help="Known users (0-499) compare against their fitted historical spending baseline. Unseen IDs use population fallback prior.",
        )
        st.number_input(
            "Transaction Amount (₹ INR)",
            min_value=1.0,
            max_value=1000000.0,
            step=500.0,
            key="sim_amount",
            help="Transaction amount in INR. Dynamically compared against user baseline to derive amount z-score.",
        )
        st.number_input(
            "Time Since Last Transaction (minutes)",
            min_value=0.0,
            max_value=10000.0,
            step=5.0,
            key="sim_time_since",
            help="Elapsed minutes since prior transaction.",
        )
        st.number_input(
            "Transaction Velocity (past 10 min)",
            min_value=0,
            max_value=50,
            step=1,
            key="sim_velocity",
            help="Number of transactions initiated in the preceding 10-minute window.",
        )
        st.selectbox(
            "Device Status",
            options=[0, 1],
            format_func=lambda x: "0 — Recognized / Familiar Device" if x == 0 else "1 — New Hardware / Unknown Fingerprint",
            key="sim_device",
            help="Flags whether the incoming request originated from a familiar device or newly seen hardware.",
        )

    with c2:
        st.number_input(
            "Geo Distance from Usual Location (km)",
            min_value=0.0,
            max_value=5000.0,
            step=10.0,
            key="sim_geo_dist",
            help="Distance in kilometers from user's historical centroid.",
        )
        st.number_input(
            "Login Burst Count (past 5 min)",
            min_value=0,
            max_value=20,
            step=1,
            key="sim_login_burst",
            help="Count of login attempts in the last 5 minutes preceding the payment.",
        )
        st.slider(
            "IP Reputation Risk Score",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="sim_ip_risk",
            help="External threat intelligence score: 0.0 (clean domestic residential) to 1.0 (TOR/high-risk proxy).",
        )
        st.slider(
            "Hour of Day (0 – 23)",
            min_value=0,
            max_value=23,
            step=1,
            key="sim_hour",
            help="Hour of transaction origination in local time (e.g. 2 = 2 AM, 14 = 2 PM).",
        )

    analyze_clicked = st.button("⚡ Analyze Transaction", type="primary", use_container_width=True)

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

    # Render Simulation Results
    if "last_sim_result" in st.session_state:
        txn_payload, result, zscore = st.session_state["last_sim_result"]
        risk_score = result["risk_score"]
        action = result["action"]
        capped = result["capped_by_safety_limit"]
        txn_id = result["transaction_id"]

        st.divider()
        st.markdown("### 📋 Risk Evaluation Results")

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Risk Score", f"{risk_score:.4f}")

        action_labels = {
            "allow": "🟢 ALLOW",
            "review": "🟡 REVIEW (Step-Up / MFA)",
            "block": "🔴 BLOCK (Immediate Hold)",
        }
        m2.metric("Decision Action", action_labels.get(action, action.upper()))
        m3.metric("Baseline Deviation (z-score)", f"{zscore:+.2f}σ")

        cap_status = "⚠️ Downgraded (Cap)" if capped else "Normal (< 50/hr)"
        m4.metric("Safety Cap Status", cap_status)

        # Visual Risk Gauge
        st.markdown(f"**Risk Score Gauge**: `{risk_score:.4f}`")
        st.progress(min(max(risk_score, 0.0), 1.0))
        st.caption(
            f"Thresholds: Allow [< {REVIEW_THRESHOLD}] | Review [{REVIEW_THRESHOLD}, {BLOCK_THRESHOLD}) | Block [≥ {BLOCK_THRESHOLD}]"
        )

        # Action banner
        if action == "allow":
            st.success(
                f"✅ **Action: ALLOW** — Risk score `{risk_score:.4f}` is below the cost-weighted threshold `{REVIEW_THRESHOLD}`. "
                "Transaction processed normally. On-demand SHAP explanations are skipped for approved transactions to preserve compute."
            )
        elif action == "review":
            if capped:
                st.warning(
                    f"⚠️ **Action: REVIEW (Downgraded by Safety Cap)** — Risk score `{risk_score:.4f}` recommended a BLOCK, "
                    f"but the hourly auto-block limit ({MAX_AUTO_BLOCKS_PER_HOUR}/hour) has been reached. "
                    "Transaction safely routed to review queue for human triage."
                )
            else:
                st.warning(
                    f"⚠️ **Action: REVIEW** — Risk score `{risk_score:.4f}` exceeds review threshold `{REVIEW_THRESHOLD}`. "
                    "Flagged for step-up multi-factor authentication (MFA) or human analyst verification."
                )
        elif action == "block":
            st.error(
                f"🛑 **Action: BLOCK** — Risk score `{risk_score:.4f}` exceeds high-confidence fraud threshold `{BLOCK_THRESHOLD}`. "
                "Transaction held immediately. Auto-block counted towards rolling safety cap."
            )

        # SHAP Explanations
        st.markdown("#### 🔍 SHAP Explainability & Risk Attribution")
        if result["top_factors"]:
            if result.get("explanation_degraded"):
                st.warning("⚠️ SHAP explanation service temporarily unavailable. Transaction was safely scored via fail-safe path.")
            else:
                st.caption("Top behavioral risk drivers contributing to this decision (TreeSHAP log-odds attribution):")
                factors_df = pd.DataFrame(result["top_factors"]).rename(columns={
                    "feature": "Feature",
                    "value": "Observed Value",
                    "contribution": "SHAP Contribution",
                    "direction": "Risk Direction",
                })
                factors_df["Risk Direction"] = factors_df["Risk Direction"].apply(
                    lambda d: "🔺 Increases Risk" if d == "increased" else "🔻 Decreases Risk"
                )
                st.dataframe(factors_df, hide_index=True)
        else:
            st.info("No SHAP factors computed (allowed transaction, compute conserved).")

        st.caption(f"📝 Audit Record Logged: `{txn_id}` at `{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}`.")


# =====================================================================
# TAB 2: Flagged Alert Feed (Batch Monitoring)
# =====================================================================
with tab_feed:
    st.subheader("Batch Flagged Transaction Feed")
    st.caption(
        f"Evaluated on 5,000 synthetic transactions | "
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
    st.markdown("##### 🚨 Flagged Queue (Select a row to inspect SHAP explanation)")

    feed = flagged.sort_values("risk_score", ascending=False)[
        ["transaction_id", "risk_score", "action", "capped_by_safety_limit", "actual_label"]
    ].reset_index(drop=True)
    feed = feed.rename(columns={
        "transaction_id": "Transaction ID",
        "risk_score": "Risk Score",
        "action": "Action",
        "capped_by_safety_limit": "Capped",
        "actual_label": "Ground Truth Fraud",
    })

    selected_idx = st.dataframe(
        feed,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    st.divider()
    st.markdown("##### 🔎 Flagged Alert Detail")
    if selected_idx["selection"]["rows"]:
        row_num = selected_idx["selection"]["rows"][0]
        selected_txn_id = feed.iloc[row_num]["Transaction ID"]
        original_idx = audit_df[audit_df["transaction_id"] == selected_txn_id].index[0]
        entry = audit_log[original_idx]

        st.markdown(
            f"**Transaction**: `{entry['transaction_id']}` | "
            f"**Risk Score**: `{entry['risk_score']}` | "
            f"**Action**: `{entry['action']}`"
        )

        if entry["top_factors"]:
            factor_df = pd.DataFrame(entry["top_factors"]).rename(columns={
                "feature": "Feature",
                "value": "Value",
                "contribution": "SHAP Contribution",
                "direction": "Effect",
            })
            st.dataframe(factor_df, hide_index=True)
        else:
            st.caption("No explanation generated for this entry.")

        if entry.get("capped_by_safety_limit"):
            st.warning(
                f"This block was downgraded to review — the {MAX_AUTO_BLOCKS_PER_HOUR}/hour "
                "safety cap had already been reached."
            )
    else:
        st.info("👆 Click any row in the flagged queue above to inspect its real-time SHAP explanation.")


# =====================================================================
# TAB 3: Audit Trail & System Telemetry
# =====================================================================
with tab_audit:
    st.subheader("System Telemetry & Operational Health")

    # Operational metrics
    t_stats = pipeline.telemetry.get_stats()
    current_blocks = pipeline.safety_cap.current_block_count
    max_blocks = pipeline.safety_cap.max_blocks_per_hour
    remaining_cap = max(0, max_blocks - current_blocks)

    tel1, tel2, tel3, tel4 = st.columns(4)
    tel1.metric("Pipeline Scored Requests", t_stats["total_requests"])
    tel2.metric("Decisions Breakdown", f"{t_stats['allow_count']} Allow / {t_stats['review_count']} Rev / {t_stats['block_count']} Blk")
    tel3.metric("Safety Cap Usage (1-hr rolling)", f"{current_blocks} / {max_blocks} used")
    tel4.metric("Remaining Block Capacity", remaining_cap)

    # Reliability and Dev Control
    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        st.caption(
            f"Reliability metrics: Degraded SHAP explanations: `{t_stats['degraded_explanation_count']}` | "
            f"Audit log I/O failures: `{t_stats['audit_failure_count']}`"
        )
    with r_col2:
        if st.button("🔄 Reset Safety Cap", use_container_width=True):
            pipeline.safety_cap.reset()
            st.success("Safety cap counter reset.")
            st.rerun()

    st.divider()
    st.subheader("Immutable Audit Trail")
    st.caption("Append-only audit log of decisions. Showing most recent 100 entries.")

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
        st.dataframe(audit_display, hide_index=True)

        st.download_button(
            "⬇️ Download Full Audit Trail (JSONL)",
            data="\n".join(json.dumps(e) for e in live_entries),
            file_name="audit_trail.jsonl",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Audit log is currently empty.")
