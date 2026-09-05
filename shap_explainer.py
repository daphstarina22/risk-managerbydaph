"""
VIGIL — Behavioral AI Risk Manager
"Detect. Explain. Decide. Protect."

SHAP explainability layer for VIGIL: explains *why* each flagged transaction
was flagged — turning a bare risk score into something a risk analyst can act on.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, we just save images
import matplotlib.pyplot as plt

from fraud_classifier import generate_synthetic_data, train_model, FEATURES


# ---------------------------------------------------------------------
# 1. Train the same model and set up a SHAP explainer
# ---------------------------------------------------------------------
def build_explainer(compute_shap_values=True):
    df = generate_synthetic_data()
    model, X_test, y_test = train_model(df)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test) if compute_shap_values else None

    return model, X_test, y_test, explainer, shap_values


# ---------------------------------------------------------------------
# 2. Global summary — which features drive the model overall
# ---------------------------------------------------------------------
def save_global_summary(shap_values, X_test, path="shap_summary.png"):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary plot to {path}")


# ---------------------------------------------------------------------
# 3. Per-alert explanation — the piece that powers the dashboard's
#    "alert detail panel" (why was THIS transaction flagged?)
# ---------------------------------------------------------------------
def explain_alert(idx, model, X_test, explainer, shap_values=None, top_n=4, verbose=True, risk_score=None):
    """
    Returns a plain-language explanation for a single flagged transaction,
    formatted the way it would appear in the dashboard's alert detail view.
    Set verbose=False to suppress printing (e.g. when called in a batch loop).
    If shap_values is None, computes local SHAP factors on-demand.
    """
    row = X_test.iloc[idx]
    if risk_score is None:
        risk_score = model.predict_proba(X_test.iloc[[idx]])[0, 1]

    if shap_values is not None:
        contributions = pd.Series(shap_values.values[idx], index=FEATURES)
    else:
        explainer_result = explainer(X_test.iloc[[idx]])
        contributions = pd.Series(explainer_result.values[0], index=FEATURES)

    top_features = contributions.abs().sort_values(ascending=False).head(top_n)

    if verbose:
        print(f"\nTransaction index {idx} — risk score {risk_score:.2f}")
        print("Top contributing factors:")
        for feat in top_features.index:
            direction = "increased" if contributions[feat] > 0 else "decreased"
            print(f"  - {feat} = {row[feat]:.2f}  ({direction} risk, "
                  f"contribution {contributions[feat]:+.3f})")

    return {
        "risk_score": round(float(risk_score), 3),
        "top_factors": [
            {
                "feature": feat,
                "value": round(float(row[feat]), 3),
                "contribution": round(float(contributions[feat]), 4),
                "direction": "increased" if contributions[feat] > 0 else "decreased",
            }
            for feat in top_features.index
        ],
    }


# ---------------------------------------------------------------------
# 4. Batch explanation for the highest-risk alerts — this is what
#    would feed the "flagged transaction feed" panel in the dashboard
# ---------------------------------------------------------------------
def explain_top_alerts(n_alerts, model, X_test, explainer, shap_values):
    probs = model.predict_proba(X_test)[:, 1]
    top_idx = np.argsort(probs)[::-1][:n_alerts]

    print(f"\n{'='*60}")
    print(f"Explaining top {n_alerts} highest-risk alerts")
    print(f"{'='*60}")

    alerts = []
    for idx in top_idx:
        alerts.append(explain_alert(idx, model, X_test, explainer, shap_values))
    return alerts


# ---------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Training model and computing SHAP values...")
    model, X_test, y_test, explainer, shap_values = build_explainer()

    save_global_summary(shap_values, X_test)

    explain_top_alerts(5, model, X_test, explainer, shap_values)
