"""
Temporal Out-of-Time (OOT) evaluation simulation for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Evaluates:
  1. True chronological stream: generates 30 days of transactions with real timestamps.
  2. Temporal Out-of-Time (OOT) Split:
     - Train on earlier transactions: Days 1–18 (60%)
     - Tune threshold on middle transactions: Days 19–24 (20%)
     - Test on latest transactions: Days 25–30 (20%)
  3. Non-stationary temporal drift:
     - End-of-month legitimate spending shifts (payday spending)
     - Fraud tactics adapt stealthily (reduced multipliers, mimicking legit signals)
  4. Comparison with Random Stratified Split on the exact same dataset to quantify
     optimistic look-ahead bias in cross-sectional evaluations.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from fraud_classifier import (
    FEATURES,
    split_data,
    cost_weighted_threshold,
)
from user_profiler import UserProfiler


# ---------------------------------------------------------------------
# 1. Chronological Data Generator
# ---------------------------------------------------------------------
def generate_temporal_data(
    n_users: int = 2000,
    n_days: int = 30,
    txns_per_day: int = 1500,
    fraud_rate: int = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic 30-day chronological transaction stream.
    Each user maintains spending and geographical baselines.
    Transactions are timestamped sequentially.

    In Days 25–30 (test period), realistic non-stationarity is introduced:
      - Legitimate spend shifts up by 15% (payday / end-of-month effect).
      - Fraudsters adapt: moderate their amount multipliers and geo jumps (stealth adaptation).
    """
    rng = np.random.default_rng(seed)
    start_date = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Per-user historical baselines
    base_amount = rng.gamma(shape=2.0, scale=500, size=n_users)

    # Track each user's last transaction timestamp
    user_last_time = {
        u: start_date + timedelta(minutes=float(rng.uniform(0, 180)))
        for u in range(n_users)
    }

    records = []
    total_txns = n_days * txns_per_day

    for day in range(n_days):
        is_drift_period = (day >= 24)  # Days 25-30: concept drift & adaptive fraud

        # Base multipliers for the period
        spend_multiplier = 1.15 if is_drift_period else 1.00
        fraud_stealth = 0.75 if is_drift_period else 1.00

        for _ in range(txns_per_day):
            u = int(rng.integers(0, n_users))
            is_fraud = bool(rng.random() < fraud_rate)

            # Advance user's time
            mean_gap = 45.0 if is_fraud else 120.0
            gap_min = float(rng.exponential(scale=mean_gap))
            current_time = user_last_time[u] + timedelta(minutes=gap_min)
            user_last_time[u] = current_time

            hour = current_time.hour
            user_base = base_amount[u] * spend_multiplier

            if is_fraud:
                # Evolving fraud: during drift period, attackers use smaller amounts & less extreme jumps
                amt_mult = float(rng.uniform(1.2, 8.0 * fraud_stealth + 1.2))
                amount = user_base * amt_mult
                txn_velocity = int(rng.poisson(2.5 * fraud_stealth))
                device_change = int(rng.choice([0, 1], p=[0.35 + (1 - fraud_stealth) * 0.2, 0.65 - (1 - fraud_stealth) * 0.2]))
                geo_dist = float(rng.gamma(shape=2.0, scale=40.0 * fraud_stealth))
                login_burst = int(rng.poisson(1.8 * fraud_stealth))
                ip_risk = float(np.clip(rng.normal(0.55 * fraud_stealth + 0.15, 0.22), 0.0, 1.0))
            else:
                amount = float(user_base * rng.lognormal(mean=0, sigma=0.4))
                txn_velocity = int(rng.poisson(0.4))
                device_change = int(rng.choice([0, 1], p=[0.92, 0.08]))
                geo_dist = float(rng.gamma(shape=1.2, scale=6.0))
                login_burst = int(rng.poisson(0.25))
                ip_risk = float(np.clip(rng.normal(0.15, 0.15), 0.0, 1.0))

            records.append({
                "timestamp": current_time,
                "day": day + 1,
                "user_id": u,
                "amount": round(amount, 2),
                "time_since_last_txn_min": round(gap_min, 1),
                "txn_velocity_10min": txn_velocity,
                "device_change": device_change,
                "geo_dist_from_usual_km": round(geo_dist, 1),
                "login_burst_count": login_burst,
                "ip_risk_score": round(ip_risk, 3),
                "hour_of_day": hour,
                "is_fraud": int(is_fraud),
            })

    df = pd.DataFrame(records)
    # Sort strictly by timestamp to maintain temporal ordering
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# 2. Evaluation Helper
# ---------------------------------------------------------------------
def evaluate_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Fits model on train, sweeps cost-optimal threshold on val, and evaluates on test.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # 1. Select optimal threshold on VALIDATION set
    val_probs = model.predict_proba(X_val)[:, 1]
    _, best = cost_weighted_threshold(y_val, val_probs)
    best_t = float(best["threshold"])

    # 2. Final evaluation on TEST set
    test_probs = model.predict_proba(X_test)[:, 1]
    preds = (test_probs >= best_t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    cost = fp * 150.0 + fn * 8000.0

    return {
        "evaluation_strategy": label,
        "pr_auc": average_precision_score(y_test, test_probs),
        "selected_threshold": best_t,
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_cost": float(cost),
    }


# ---------------------------------------------------------------------
# 3. Main Temporal Experiment
# ---------------------------------------------------------------------
def run_temporal_comparison(seed: int = 42) -> pd.DataFrame:
    """
    Compares Random Stratified Split vs. Temporal Out-of-Time (OOT) Split
    on the exact same 30-day chronological dataset.
    """
    print("Generating 30-day chronological transaction stream (n=45,000)...")
    temporal_df = generate_temporal_data(n_users=2000, n_days=30, txns_per_day=1500, seed=seed)

    # -----------------------------------------------------------------
    # A. Strategy 1: Random Stratified Split (Standard Cross-Sectional)
    # -----------------------------------------------------------------
    train_rand, val_rand, test_rand = split_data(
        temporal_df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_state=seed
    )

    prof_rand = UserProfiler().fit(train_rand)
    X_train_rand = prof_rand.transform(train_rand)[FEATURES]
    y_train_rand = train_rand["is_fraud"]
    X_val_rand = prof_rand.transform(val_rand)[FEATURES]
    y_val_rand = val_rand["is_fraud"]
    X_test_rand = prof_rand.transform(test_rand)[FEATURES]
    y_test_rand = test_rand["is_fraud"]

    rand_result = evaluate_split(
        X_train_rand, y_train_rand, X_val_rand, y_val_rand, X_test_rand, y_test_rand,
        label="Random Stratified Split (Cross-Sectional)", seed=seed
    )

    # -----------------------------------------------------------------
    # B. Strategy 2: Temporal Out-of-Time (OOT) Split (Chronological)
    #    - Train: Days 1–18 (earlier transactions)
    #    - Val:   Days 19–24 (middle transactions)
    #    - Test:  Days 25–30 (latest transactions under drift)
    # -----------------------------------------------------------------
    train_oot = temporal_df[temporal_df["day"] <= 18].copy()
    val_oot = temporal_df[(temporal_df["day"] > 18) & (temporal_df["day"] <= 24)].copy()
    test_oot = temporal_df[temporal_df["day"] > 24].copy()

    prof_oot = UserProfiler().fit(train_oot)
    X_train_oot = prof_oot.transform(train_oot)[FEATURES]
    y_train_oot = train_oot["is_fraud"]
    X_val_oot = prof_oot.transform(val_oot)[FEATURES]
    y_val_oot = val_oot["is_fraud"]
    X_test_oot = prof_oot.transform(test_oot)[FEATURES]
    y_test_oot = test_oot["is_fraud"]

    oot_result = evaluate_split(
        X_train_oot, y_train_oot, X_val_oot, y_val_oot, X_test_oot, y_test_oot,
        label="Temporal Out-of-Time Split (Chronological OOT)", seed=seed
    )

    comparison_df = pd.DataFrame([rand_result, oot_result])
    return comparison_df


if __name__ == "__main__":
    results = run_temporal_comparison(seed=42)
    print("\n" + "=" * 90)
    print("TEMPORAL OUT-OF-TIME (OOT) VS. RANDOM SPLIT COMPARISON")
    print("=" * 90)
    print(results[[
        "evaluation_strategy", "pr_auc", "selected_threshold", "precision", "recall",
        "false_positives", "false_negatives", "total_cost"
    ]].to_string(index=False))

    rand_auc = results.loc[results["evaluation_strategy"].str.contains("Random"), "pr_auc"].values[0]
    oot_auc = results.loc[results["evaluation_strategy"].str.contains("Temporal"), "pr_auc"].values[0]
    auc_drop = rand_auc - oot_auc

    print(f"\nTemporal Non-Stationarity Degradation: {auc_drop:+.4f} PR-AUC")
    print("Core Engineering Insights:")
    print("  1. Random stratified splitting leaks future data distributions into the training set,")
    print("     producing overly optimistic PR-AUC and masking real-world drift.")
    print("  2. Temporal OOT evaluation honestly simulates deployment: training on historical data,")
    print("     tuning thresholds on recent validation data, and evaluating on future transactions.")
    print("  3. Recommendation for production: Retrain the XGBoost model on a rolling 2–4 week window")
    print("     and track PR-AUC on weekly OOT test holdouts to trigger proactive retraining.")
