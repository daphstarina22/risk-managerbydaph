"""
Fraud-spike / account-takeover detector — Streamlit dashboard.
Razorpay Buildathon — AI Risk Manager track.

Wraps fraud_classifier.py, shap_explainer.py, and decision_layer.py in a
clickable UI: summary metrics, a flagged transaction feed, per-alert SHAP
explanations, and a live audit trail.
"""

import json

import pandas as pd
import streamlit as st

from fraud_classifier import FEATURES
from shap_explainer import build_explainer, explain_alert
from decision_layer import process_batch, REVIEW_THRESHOLD, BLOCK_THRESHOLD, MAX_AUTO_BLOCKS_PER_HOUR


st.set_page_config(page_title="Fraud-spike monitor", layout="wide")


# ---------------------------------------------------------------------
# Train once per session, not on every interaction
# ---------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model, X_test, y_test, explainer, shap_values = build_explainer()
    audit_log = process_batch(model, X_test, y_test, explainer, shap_values)
    return model, X_test, y_test, explainer, shap_values, audit_log


with st.spinner("Training model and running the decision pipeline..."):
    model, X_test, y_test, explainer, shap_values, audit_log = load_pipeline()

audit_df = pd.DataFrame(audit_log)
probs = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Fraud-spike / account-takeover monitor")
st.caption("Razorpay AI Buildathon — AI Risk Manager track")


# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------
flagged = audit_df[audit_df["action"].isin(["review", "block"])]
precision_on_flagged = flagged["actual_label"].mean() if len(flagged) else 0
missed_fraud = len(audit_df[(audit_df["action"] == "allow") & (audit_df["actual_label"] == 1)])
capped_count = audit_df["capped_by_safety_limit"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision on flagged", f"{precision_on_flagged:.1%}")
col2.metric("Fraud missed", missed_fraud)
col3.metric("Blocked", len(audit_df[audit_df["action"] == "block"]))
col4.metric("Held by safety cap", int(capped_count))

st.divider()


# ---------------------------------------------------------------------
# Flagged transaction feed
# ---------------------------------------------------------------------
st.subheader("Flagged transactions")
st.caption(f"Rules: review \u2265 {REVIEW_THRESHOLD}, block \u2265 {BLOCK_THRESHOLD}, "
           f"max {MAX_AUTO_BLOCKS_PER_HOUR} auto-blocks/hour")

feed = flagged.sort_values("risk_score", ascending=False)[
    ["transaction_id", "risk_score", "action", "capped_by_safety_limit", "actual_label"]
].reset_index(drop=True)
feed = feed.rename(columns={
    "transaction_id": "Transaction",
    "risk_score": "Risk score",
    "action": "Action",
    "capped_by_safety_limit": "Capped",
    "actual_label": "Actually fraud",
})

selected_idx = st.dataframe(
    feed, use_container_width=True, hide_index=True,
    on_select="rerun", selection_mode="single-row",
)

st.divider()


# ---------------------------------------------------------------------
# Alert detail — SHAP explanation for the selected transaction
# ---------------------------------------------------------------------
st.subheader("Alert detail")

if selected_idx["selection"]["rows"]:
    row_num = selected_idx["selection"]["rows"][0]
    txn_id = feed.iloc[row_num]["Transaction"]
    original_idx = audit_df[audit_df["transaction_id"] == txn_id].index[0]

    entry = audit_log[original_idx]
    st.markdown(f"**{entry['transaction_id']}** — risk score `{entry['risk_score']}` "
                f"— action: **{entry['action']}**")

    if entry["top_factors"]:
        factor_df = pd.DataFrame(entry["top_factors"]).rename(columns={
            "feature": "Feature", "value": "Value",
            "contribution": "SHAP contribution", "direction": "Effect",
        })
        st.dataframe(factor_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No explanation generated for allowed transactions (by design, to save compute).")

    if entry["capped_by_safety_limit"]:
        st.warning(f"This block was downgraded to review — the {MAX_AUTO_BLOCKS_PER_HOUR}/hour "
                   f"safety cap had already been reached.")
else:
    st.caption("Select a row in the table above to see why it was flagged.")

st.divider()


# ---------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------
st.subheader("Audit trail")
st.caption("Every decision, logged. Showing the most recent 100 entries.")

audit_display = audit_df.tail(100)[["timestamp", "transaction_id", "risk_score", "action", "capped_by_safety_limit"]]
st.dataframe(audit_display, use_container_width=True, hide_index=True)

st.download_button(
    "Download full audit trail (JSONL)",
    data="\n".join(json.dumps(e) for e in audit_log),
    file_name="audit_trail.jsonl",
    mime="application/json",
)
