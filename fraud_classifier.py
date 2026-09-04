"""
Fraud-spike / account-takeover detector — core classifier.
Razorpay Buildathon — AI Risk Manager track.

Pipeline: synthetic data -> feature engineering -> XGBoost classifier
-> precision/recall/PR-AUC -> cost-weighted threshold selection.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

from user_profiler import UserProfiler


# ---------------------------------------------------------------------
# 1. Synthetic data generation (deterministic with scoped RNG)
# ---------------------------------------------------------------------
def generate_synthetic_data(n_users=2000, n_txns=40000, fraud_rate=0.015, seed=42):
    """
    Generates transactions with per-user 'normal' behavior baselines,
    then injects fraud as deviations from that baseline (device change,
    velocity spikes, geo jumps, login bursts) — not random noise.
    Uses a local/scoped random generator to ensure exact determinism and reproducibility.
    """
    rng = np.random.default_rng(seed)

    user_ids = rng.integers(0, n_users, size=n_txns)
    base_amount = rng.gamma(shape=2.0, scale=500, size=n_users)       # each user's typical spend
    base_geo = rng.uniform(0, 100, size=(n_users, 2))                  # Preserved to maintain exact deterministic RNG sequence reproducibility

    rows = []
    n_fraud = int(n_txns * fraud_rate)
    fraud_idx = set(rng.choice(n_txns, size=n_fraud, replace=False))

    for i in range(n_txns):
        u = user_ids[i]
        is_fraud = i in fraud_idx

        if is_fraud:
            # fraud = *usually* deviates from baseline, but overlaps with legit
            # behavior often enough that no single feature perfectly separates it
            amount = base_amount[u] * rng.uniform(1.2, 8)
            txn_velocity_10min = rng.poisson(2.5)
            device_change = rng.choice([0, 1], p=[0.35, 0.65])          # not always
            geo_dist = rng.gamma(shape=2.0, scale=40)                   # heavy overlap with legit tail
            login_burst = rng.poisson(1.8)
            ip_risk_score = np.clip(rng.normal(0.55, 0.22), 0, 1)
            hour = rng.integers(0, 24)                                  # fraud happens anytime
        else:
            amount = base_amount[u] * rng.lognormal(mean=0, sigma=0.4)
            txn_velocity_10min = rng.poisson(0.4)
            device_change = rng.choice([0, 1], p=[0.92, 0.08])
            geo_dist = rng.gamma(shape=1.2, scale=6)
            login_burst = rng.poisson(0.25)
            ip_risk_score = np.clip(rng.normal(0.15, 0.15), 0, 1)
            hour = rng.integers(0, 24)

        time_since_last = rng.exponential(scale=90 if not is_fraud else 35)

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
def engineer_features(df, profiler=None):
    """
    Feature engineering using UserProfiler.
    If profiler is provided, uses learned training baselines.
    If profiler is None, fits on df directly (legacy helper).
    """
    if profiler is None:
        profiler = UserProfiler()
        profiler.fit(df)
    return profiler.transform(df)


FEATURES = [
    "amount", "amount_zscore", "time_since_last_txn_min", "txn_velocity_10min",
    "device_change", "geo_dist_from_usual_km", "login_burst_count",
    "ip_risk_score", "hour_of_day",
]


# ---------------------------------------------------------------------
# 3. Stratified 3-way data splitting: Train (60%) / Val (20%) / Test (20%)
# ---------------------------------------------------------------------
def split_data(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    Splits DataFrame into strictly non-overlapping, stratified partitions:
      - Train set (default 60%): used exclusively for baseline fitting and model training.
      - Validation set (default 20%): used exclusively for threshold optimization.
      - Test set (default 20%): held untouched until final evaluation.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    stratify_col = df["is_fraud"] if "is_fraud" in df.columns else None

    temp_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, test_size=temp_size, stratify=stratify_col, random_state=random_state
    )

    val_temp_ratio = val_ratio / temp_size
    stratify_temp = temp_df["is_fraud"] if stratify_col is not None else None
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_temp_ratio), stratify=stratify_temp, random_state=random_state
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------
# 4. Model training with split-safe UserProfiler and class weighting
# ---------------------------------------------------------------------
def train_model(df, profiler=None, return_val=False, random_state=42):
    """
    Splits data first into Train (60%), Validation (20%), and Test (20%).
    Fits UserProfiler strictly on the training set, then transforms train, val, and test.
    Trains XGBoost exclusively on the training set.

    If return_val is True:
        returns model, X_val, y_val, X_test, y_test
    If return_val is False (default for backwards compatibility):
        returns model, X_test, y_test
    """
    if "amount_zscore" not in df.columns:
        train_df, val_df, test_df = split_data(df, random_state=random_state)

        if profiler is None:
            profiler = UserProfiler()
            profiler.fit(train_df)

        train_df = profiler.transform(train_df)
        val_df = profiler.transform(val_df)
        test_df = profiler.transform(test_df)

        X_train, y_train = train_df[FEATURES], train_df["is_fraud"]
        X_val, y_val = val_df[FEATURES], val_df["is_fraud"]
        X_test, y_test = test_df[FEATURES], test_df["is_fraud"]
    else:
        # Legacy path if amount_zscore is already present
        train_df, val_df, test_df = split_data(df, random_state=random_state)
        if profiler is None and "user_id" in df.columns:
            profiler = UserProfiler().fit(train_df)
        X_train, y_train = train_df[FEATURES], train_df["is_fraud"]
        X_val, y_val = val_df[FEATURES], val_df["is_fraud"]
        X_test, y_test = test_df[FEATURES], test_df["is_fraud"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    model.profiler = profiler

    if return_val:
        return model, X_val, y_val, X_test, y_test
    return model, X_test, y_test


# ---------------------------------------------------------------------
# 5. Evaluation — precision, recall, PR-AUC, F1, and financial cost
# ---------------------------------------------------------------------
def evaluate(model, X_test, y_test, threshold=0.5, label="Test"):
    """
    Evaluates model predictions against precision, recall, PR-AUC, F1, and cost.
    Accepts an operating threshold to evaluate binary classification decisions.
    """
    probs = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, probs)
    preds = (probs >= threshold).astype(int)

    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    total_cost = fp * 150 + fn * 8000

    print(f"[{label} Evaluation @ threshold {threshold:.2f}]")
    print(f"  PR-AUC (average precision):        {ap:.4f}")
    print(f"  Precision @ {threshold:.2f} threshold:          {prec:.4f}")
    print(f"  Recall @ {threshold:.2f} threshold:             {rec:.4f}")
    print(f"  F1 @ {threshold:.2f} threshold:                 {f1:.4f}")
    print(f"  Confusion Matrix:                 TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Estimated Total Cost:             Rs {total_cost:,.0f}")

    return probs


# ---------------------------------------------------------------------
# 6. Cost-weighted threshold selection — optimized on VALIDATION set
# ---------------------------------------------------------------------
def cost_weighted_threshold(y_val, probs, cost_fp=150, cost_fn=8000):
    """
    cost_fp: cost of wrongly blocking a legitimate transaction
             (support ticket, churn risk, lost goodwill) in rupees.
    cost_fn: cost of missing actual fraud (average fraud loss) in rupees.
    Sweeps thresholds on VALIDATION set and reports the one minimizing total expected cost.
    Must be called on the validation set, NEVER on the final test set.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
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

    print("\nCost-weighted threshold sweep on Validation set (top 5 by lowest cost):")
    print(results_df.head(5).to_string(index=False))
    print(f"\n>>> Recommended operating threshold: {best['threshold']} "
          f"(Validation estimated cost: Rs {best['total_cost_inr']:,.0f})")

    return results_df, best


# ---------------------------------------------------------------------
# 7. Model artifact persistence (offline export & load)
# ---------------------------------------------------------------------
def export_artifacts(model, profiler=None, artifact_dir="artifacts"):
    """
    Saves trained XGBoost model and learned user baselines to disk.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    model_path = os.path.join(artifact_dir, "model.json")
    model.save_model(model_path)

    if profiler is None and hasattr(model, "profiler"):
        profiler = model.profiler

    if profiler is not None:
        baselines_path = os.path.join(artifact_dir, "user_baselines.json")
        profiler.save(baselines_path)

    print(f"Exported model artifacts to '{artifact_dir}/'")
    return model_path


def load_artifacts(artifact_dir="artifacts"):
    """
    Loads pre-trained XGBoost model and learned user baselines from disk.
    """
    model_path = os.path.join(artifact_dir, "model.json")
    baselines_path = os.path.join(artifact_dir, "user_baselines.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    model = XGBClassifier()
    model.load_model(model_path)

    if os.path.exists(baselines_path):
        profiler = UserProfiler.load(baselines_path)
    else:
        profiler = UserProfiler()

    model.profiler = profiler
    return model, profiler


# ---------------------------------------------------------------------
# Run the full pipeline
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic transaction data (deterministic seed=42)...")
    df = generate_synthetic_data(seed=42)
    print(f"Total transactions: {len(df)}  |  Fraud rate: {df['is_fraud'].mean():.3%}\n")

    print("Training XGBoost classifier with strict 3-way split (Train 60% / Val 20% / Test 20%)...")
    model, X_val, y_val, X_test, y_test = train_model(df, return_val=True)

    print("\n" + "="*60)
    print("STEP 1: VALIDATION EVALUATION & THRESHOLD OPTIMIZATION")
    print("="*60)
    val_probs = model.predict_proba(X_val)[:, 1]
    results_df, best = cost_weighted_threshold(y_val, val_probs)
    operating_threshold = float(best["threshold"])

    print(f"\nEvaluating Validation set at default 0.5 threshold:")
    _ = evaluate(model, X_val, y_val, threshold=0.5, label="Validation (Default 0.5)")

    print(f"\nEvaluating Validation set at locked cost-optimal threshold ({operating_threshold:.2f}):")
    _ = evaluate(model, X_val, y_val, threshold=operating_threshold, label=f"Validation (Optimal {operating_threshold:.2f})")

    print("\n" + "="*60)
    print(f"STEP 2: FINAL UNTOUCHED TEST EVALUATION (Locked threshold = {operating_threshold:.2f})")
    print("="*60)
    print(f"\nEvaluating untouched Test set at default 0.5 threshold:")
    _ = evaluate(model, X_test, y_test, threshold=0.5, label="Test (Default 0.5)")

    print(f"\nEvaluating untouched Test set at locked validation threshold ({operating_threshold:.2f}):")
    probs = evaluate(model, X_test, y_test, threshold=operating_threshold, label=f"Test (Locked {operating_threshold:.2f})")

    # Feature importance — useful for your explainability layer / pitch
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    # Export artifacts for offline inference
    export_artifacts(model, getattr(model, "profiler", None))
