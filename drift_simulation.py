"""
Data drift simulation for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Trains the model on "today's" data, then evaluates it on a simulated
"month later" batch where user spending habits and fraud tactics have
shifted. A model that degrades under drift — and is honest about it — is
more trustworthy than one that implicitly claims to work forever.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

from fraud_classifier import engineer_features, FEATURES

RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------
# Generate a drifted batch: spending habits shift up, fraud tactics adapt
# ---------------------------------------------------------------------
def generate_drifted_data(n_users=2000, n_txns=40000, fraud_rate=0.015,
                           spend_drift=1.35, fraud_stealth=0.7):
    """
    spend_drift:   legitimate users' typical spend increases over time
                   (inflation, seasonal spending, habit change) — makes
                   amount-based features less discriminative if unadjusted.
    fraud_stealth: fraud tactics partially adapt to look more like normal
                   behavior (e.g. smaller amount multiplier, less extreme
                   geo jumps) — the classic "arms race" dynamic in fraud.
    """
    user_ids = RNG.integers(0, n_users, size=n_txns)
    base_amount = RNG.gamma(shape=2.0, scale=500 * spend_drift, size=n_users)

    rows = []
    n_fraud = int(n_txns * fraud_rate)
    fraud_idx = set(RNG.choice(n_txns, size=n_fraud, replace=False))

    for i in range(n_txns):
        u = user_ids[i]
        is_fraud = i in fraud_idx

        if is_fraud:
            amount = base_amount[u] * RNG.uniform(1.2, 8 * fraud_stealth + 1.2)
            txn_velocity_10min = RNG.poisson(2.5 * fraud_stealth)
            device_change = RNG.choice([0, 1], p=[0.35 + (1 - fraud_stealth) * 0.2,
                                                    0.65 - (1 - fraud_stealth) * 0.2])
            geo_dist = RNG.gamma(shape=2.0, scale=40 * fraud_stealth)
            login_burst = RNG.poisson(1.8 * fraud_stealth)
            ip_risk_score = np.clip(RNG.normal(0.55 * fraud_stealth + 0.15, 0.22), 0, 1)
            hour = RNG.integers(0, 24)
        else:
            amount = base_amount[u] * RNG.lognormal(mean=0, sigma=0.4)
            txn_velocity_10min = RNG.poisson(0.4)
            device_change = RNG.choice([0, 1], p=[0.92, 0.08])
            geo_dist = RNG.gamma(shape=1.2, scale=6)
            login_burst = RNG.poisson(0.25)
            ip_risk_score = np.clip(RNG.normal(0.15, 0.15), 0, 1)
            hour = RNG.integers(0, 24)

        time_since_last = RNG.exponential(scale=90 if not is_fraud else 35)

        rows.append({
            "user_id": u, "amount": round(amount, 2),
            "time_since_last_txn_min": round(time_since_last, 1),
            "txn_velocity_10min": txn_velocity_10min, "device_change": device_change,
            "geo_dist_from_usual_km": round(geo_dist, 1), "login_burst_count": login_burst,
            "ip_risk_score": round(ip_risk_score, 3), "hour_of_day": hour,
            "is_fraud": int(is_fraud),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Compare: model trained on "today" evaluated on "today" vs "month later"
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from fraud_classifier import generate_synthetic_data

    print("Training model on 'today' data...")
    today_df = engineer_features(generate_synthetic_data())
    X = today_df[FEATURES]
    y = today_df["is_fraud"]
    X_train, X_test_today, y_train, y_test_today = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)

    def evaluate(X_eval, y_eval, label):
        probs = model.predict_proba(X_eval)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "label": label,
            "pr_auc": average_precision_score(y_eval, probs),
            "precision": precision_score(y_eval, preds),
            "recall": recall_score(y_eval, preds),
        }

    today_result = evaluate(X_test_today, y_test_today, "Today (no drift)")

    print("Generating 'one month later' drifted batch...")
    drifted_df = engineer_features(generate_drifted_data())
    X_drift = drifted_df[FEATURES]
    y_drift = drifted_df["is_fraud"]

    drift_result = evaluate(X_drift, y_drift, "One month later (drifted, same model)")

    comparison = pd.DataFrame([today_result, drift_result]).set_index("label")
    print("\n" + "=" * 60)
    print("Model performance: today vs. one month later (no retraining)")
    print("=" * 60)
    print(comparison.round(4).to_string())

    precision_drop = today_result["precision"] - drift_result["precision"]
    recall_drop = today_result["recall"] - drift_result["recall"]

    print(f"\nPrecision drop: {precision_drop:+.4f}")
    print(f"Recall drop:    {recall_drop:+.4f}")

    if precision_drop > 0.02 or recall_drop > 0.02:
        print("\n>>> Meaningful degradation detected under simulated drift.")
        print(">>> Recommendation: retrain on a rolling 2-4 week window, and monitor")
        print(">>> precision/recall on a held-out recent slice weekly, alerting if either")
        print(">>> metric drops more than 2 points from its trained baseline.")
    else:
        print("\n>>> Degradation is small in this simulation. Still recommend monthly")
        print(">>> retraining as a baseline cadence, since real-world drift and adaptive")
        print(">>> fraud tactics are likely to be less forgiving than this simulation.")
