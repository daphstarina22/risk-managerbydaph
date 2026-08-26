"""
Fraud-spike / account-takeover detector — core classifier.
Razorpay Buildathon — AI Risk Manager track.

Pipeline: synthetic data -> feature engineering -> XGBoost classifier
-> precision/recall/PR-AUC -> cost-weighted threshold selection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------
def generate_synthetic_data(n_users=2000, n_txns=40000, fraud_rate=0.015):
    """
    Generates transactions with per-user 'normal' behavior baselines,
    then injects fraud as deviations from that baseline (device change,
    velocity spikes, geo jumps, login bursts) — not random noise.
    """
    user_ids = RNG.integers(0, n_users, size=n_txns)
    base_amount = RNG.gamma(shape=2.0, scale=500, size=n_users)       # each user's typical spend
    base_geo = RNG.uniform(0, 100, size=(n_users, 2))                  # each user's usual lat/lon proxy

    rows = []
    n_fraud = int(n_txns * fraud_rate)
    fraud_idx = set(RNG.choice(n_txns, size=n_fraud, replace=False))

    for i in range(n_txns):
        u = user_ids[i]
        is_fraud = i in fraud_idx

        if is_fraud:
            # fraud = *usually* deviates from baseline, but overlaps with legit
            # behavior often enough that no single feature perfectly separates it
            amount = base_amount[u] * RNG.uniform(1.2, 8)
            txn_velocity_10min = RNG.poisson(2.5)
            device_change = RNG.choice([0, 1], p=[0.35, 0.65])          # not always
            geo_dist = RNG.gamma(shape=2.0, scale=40)                   # heavy overlap with legit tail
            login_burst = RNG.poisson(1.8)
            ip_risk_score = np.clip(RNG.normal(0.55, 0.22), 0, 1)
            hour = RNG.integers(0, 24)                                  # fraud happens anytime
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
            "user_id": u,
            "amount": round(amount, 2),
            "time_since_last_txn_min": round(time_since_last, 1),
            "txn_velocity_10min": txn_velocity_10min,
            "device_change": device_change,
            "geo_dist_from_usual_km": round(geo_dist, 1),
            "login_burst_count": login_burst,
            "ip_risk_score": round(ip_risk_score, 3),
            "hour_of_day": hour,
            "is_fraud": int(is_fraud),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Feature engineering — deviation-from-normal, not raw values
# ---------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).fillna(1e-3)
    df = df.join(user_stats, on="user_id", rsuffix="_user")
    df["amount_zscore"] = (df["amount"] - df["mean"]) / df["std"].replace(0, 1e-3)
    df.drop(columns=["mean", "std"], inplace=True)
    return df


FEATURES = [
    "amount", "amount_zscore", "time_since_last_txn_min", "txn_velocity_10min",
    "device_change", "geo_dist_from_usual_km", "login_burst_count",
    "ip_risk_score", "hour_of_day",
]


# ---------------------------------------------------------------------
# 3 & 4. Split + train with class weighting
# ---------------------------------------------------------------------
def train_model(df):
    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ---------------------------------------------------------------------
# 5. Evaluation — precision, recall, PR-AUC (never accuracy)
# ---------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, probs)
    preds_default = (probs >= 0.5).astype(int)

    print(f"PR-AUC (average precision):        {ap:.3f}")
    print(f"Precision @ 0.5 threshold:          {precision_score(y_test, preds_default):.3f}")
    print(f"Recall @ 0.5 threshold:              {recall_score(y_test, preds_default):.3f}")
    print(f"F1 @ 0.5 threshold:                  {f1_score(y_test, preds_default):.3f}")
    return probs


# ---------------------------------------------------------------------
# 6. Cost-weighted threshold selection — the differentiating piece
# ---------------------------------------------------------------------
def cost_weighted_threshold(y_test, probs, cost_fp=150, cost_fn=8000):
    """
    cost_fp: cost of wrongly blocking a legitimate transaction
             (support ticket, churn risk, lost goodwill) in rupees.
    cost_fn: cost of missing actual fraud (average fraud loss) in rupees.
    Sweeps thresholds and reports the one minimizing total expected cost.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        total_cost = fp * cost_fp + fn * cost_fn
        results.append({
            "threshold": round(t, 2),
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "total_cost_inr": total_cost,
        })

    results_df = pd.DataFrame(results).sort_values("total_cost_inr")
    best = results_df.iloc[0]

    print("\nCost-weighted threshold sweep (top 5 by lowest total cost):")
    print(results_df.head(5).to_string(index=False))
    print(f"\n>>> Recommended threshold: {best['threshold']} "
          f"(estimated cost: Rs {best['total_cost_inr']:,.0f} on this test batch)")

    return results_df, best


# ---------------------------------------------------------------------
# Run the full pipeline
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic transaction data...")
    df = generate_synthetic_data()
    print(f"Total transactions: {len(df)}  |  Fraud rate: {df['is_fraud'].mean():.3%}\n")

    df = engineer_features(df)

    print("Training XGBoost classifier...")
    model, X_test, y_test = train_model(df)

    print("\nEvaluation on held-out test set:")
    probs = evaluate(model, X_test, y_test)

    cost_weighted_threshold(y_test, probs)

    # Feature importance — useful for your explainability layer / pitch
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())
