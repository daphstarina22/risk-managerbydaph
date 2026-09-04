"""
Additional evaluation rigor for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Evaluations:
  1. Precision-recall curve plot (visual, for README/pitch)
  2. K-fold cross-validation (leak-free fold baselines)
  3. Analyst workload projection (alerts/day a human would need to review)
  4. Operational Payment Metrics: Recall @ 1% FPR and Precision@Top-K
  5. Sensor / Telemetry Dropout Robustness (graceful degradation under missing features)
  6. Probability Calibration & Brier Score (validates risk scores as true posteriors)
"""

from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from fraud_classifier import (
    FEATURES,
    generate_synthetic_data,
    train_model,
)
from user_profiler import UserProfiler


# ---------------------------------------------------------------------
# 1. Precision-recall curve plot
# ---------------------------------------------------------------------
def plot_pr_curve(y_test, probs, path="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve (PR-AUC = {ap:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved PR curve to {path}")


# ---------------------------------------------------------------------
# 2. K-fold cross-validation (leak-free fold baselines)
# ---------------------------------------------------------------------
def cross_validate(df, n_splits=5):
    y = df["is_fraud"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y), 1):
        train_split = df.iloc[train_idx]
        test_split = df.iloc[test_idx]

        if "amount_zscore" not in df.columns:
            profiler = UserProfiler().fit(train_split)
            X_train = profiler.transform(train_split)[FEATURES]
            X_test = profiler.transform(test_split)[FEATURES]
        else:
            X_train = train_split[FEATURES]
            X_test = test_split[FEATURES]

        y_train = train_split["is_fraud"]
        y_test = test_split["is_fraud"]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.05).astype(int)

        fold_results.append({
            "fold": fold_idx,
            "pr_auc": average_precision_score(y_test, probs),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
        })

    results_df = pd.DataFrame(fold_results)
    print(f"\n{n_splits}-fold cross-validation results (leak-free fold baselines @ 0.05 threshold):")
    print(results_df.round(4).to_string(index=False))
    print(f"\nMean PR-AUC:    {results_df['pr_auc'].mean():.4f}  (+/- {results_df['pr_auc'].std():.4f})")
    print(f"Mean Precision: {results_df['precision'].mean():.4f}  (+/- {results_df['precision'].std():.4f})")
    print(f"Mean Recall:    {results_df['recall'].mean():.4f}  (+/- {results_df['recall'].std():.4f})")

    return results_df


# ---------------------------------------------------------------------
# 3. Analyst workload projection
# ---------------------------------------------------------------------
def project_analyst_workload(n_flagged_in_test, test_size, daily_txn_volume, avg_review_minutes=3):
    """
    Scales the flagged rate observed on the test set up to an assumed daily
    transaction volume, to answer: 'how many alerts would a human review per day?'
    """
    flagged_rate = n_flagged_in_test / test_size
    daily_alerts = flagged_rate * daily_txn_volume
    daily_review_hours = (daily_alerts * avg_review_minutes) / 60

    print(f"\nAnalyst workload projection (assuming {daily_txn_volume:,} transactions/day):")
    print(f"  Flagged rate observed: {flagged_rate:.4%}")
    print(f"  Projected alerts/day:  {daily_alerts:,.0f}")
    print(f"  Analyst-hours/day @ {avg_review_minutes} min/review: {daily_review_hours:,.1f} hours")
    print(f"  -> roughly {daily_review_hours/8:.1f} full-time analysts needed at this alert rate")

    return {
        "flagged_rate": flagged_rate,
        "daily_alerts": daily_alerts,
        "daily_review_hours": daily_review_hours,
    }


# ---------------------------------------------------------------------
# 4. Operational Payment Metrics (Recall @ Fixed FPR & Precision@K)
# ---------------------------------------------------------------------
def recall_at_fixed_fpr(y_true: np.ndarray, probs: np.ndarray, target_fpr: float = 0.01) -> Dict[str, Any]:
    """
    Finds the operating threshold where False Positive Rate <= target_fpr (e.g. 1.0%),
    and computes the achievable fraud Recall at that strict threshold.
    Crucial for payments where merchant checkout drop-off must be strictly capped.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    valid_idx = np.where(fpr <= target_fpr)[0]
    best_idx = valid_idx[-1] if len(valid_idx) > 0 else 0

    achieved_fpr = float(fpr[best_idx])
    achieved_recall = float(tpr[best_idx])
    threshold = float(thresholds[best_idx])

    return {
        "target_fpr": target_fpr,
        "achieved_fpr": round(achieved_fpr, 4),
        "threshold": round(threshold, 4),
        "recall_at_target_fpr": round(achieved_recall, 4),
    }


def precision_at_k(y_true: np.ndarray, probs: np.ndarray, k_values: List[int] = [50, 100, 200]) -> pd.DataFrame:
    """
    Computes precision among the top-K highest predicted risk transactions.
    Directly models fixed daily human review queue capacity.
    """
    df_eval = pd.DataFrame({"y_true": np.array(y_true), "prob": np.array(probs)}).sort_values("prob", ascending=False)
    results = []
    total_fraud = int(df_eval["y_true"].sum())

    for k in k_values:
        top_k = df_eval.head(k)
        tp = int(top_k["y_true"].sum())
        prec_k = tp / k if k > 0 else 0.0
        recall_k = tp / total_fraud if total_fraud > 0 else 0.0
        results.append({
            "k (Top Alerts Queue)": k,
            "true_positives": tp,
            "false_positives": k - tp,
            "precision_at_k": round(prec_k, 4),
            "recall_at_k": round(recall_k, 4),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# 5. Sensor / Telemetry Dropout Robustness
# ---------------------------------------------------------------------
def evaluate_feature_dropout(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Tests model robustness when upstream sensors fail or are blocked:
      - Baseline (all features available)
      - Geolocation Missing (geo_dist = 0)
      - IP Intelligence Timeout (ip_risk = 0.5 uninformative prior)
      - Device Fingerprint Unavailable (device_change = 0)
      - Mobile Sensor Outage (geo = 0 and ip = 0.5)
    """
    scenarios = {
        "Baseline (All Sensors)": X_test.copy(),
        "Missing Geolocation (geo=0)": X_test.assign(geo_dist_from_usual_km=0.0),
        "IP Intel Timeout (ip_risk=0.5)": X_test.assign(ip_risk_score=0.5),
        "Device Fingerprint Missing (dev=0)": X_test.assign(device_change=0),
        "Degraded Sensors (geo=0 + ip=0.5)": X_test.assign(geo_dist_from_usual_km=0.0, ip_risk_score=0.5),
    }

    results = []
    base_probs = model.predict_proba(X_test)[:, 1]
    base_auc = average_precision_score(y_test, base_probs)

    for name, X_scen in scenarios.items():
        probs = model.predict_proba(X_scen)[:, 1]
        preds = (probs >= 0.05).astype(int)
        auc = average_precision_score(y_test, probs)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        results.append({
            "telemetry_scenario": name,
            "pr_auc": round(auc, 4),
            "auc_retention": f"{(auc / base_auc) * 100:.1f}%",
            "precision_at_0.05": round(prec, 4),
            "recall_at_0.05": round(rec, 4),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# 6. Probability Calibration & Brier Score
# ---------------------------------------------------------------------
def evaluate_calibration(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Measures probability calibration quality:
      - Brier Score: mean squared error of predicted probabilities.
      - Expected Calibration Error (ECE): weighted average gap between confidence and empirical frequency.
    """
    y_arr = np.array(y_true)
    p_arr = np.array(probs)
    brier = brier_score_loss(y_arr, p_arr)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(p_arr, bins) - 1
    ece = 0.0
    n = len(p_arr)

    for i in range(n_bins):
        in_bin = (bin_assignments == i)
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(y_arr[in_bin]))
            bin_conf = float(np.mean(p_arr[in_bin]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return {
        "brier_score": round(float(brier), 6),
        "expected_calibration_error": round(float(ece), 4),
    }


# ---------------------------------------------------------------------
# Run all evaluations
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic data (n=40,000)...")
    df = generate_synthetic_data(n_txns=40000, seed=42)

    # Train model using split-safe UserProfiler
    model, X_test, y_test = train_model(df)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- 1. Precision-Recall curve ---")
    plot_pr_curve(y_test, probs)

    print("\n--- 2. Cross-validation (leak-free) ---")
    cross_validate(df, n_splits=5)

    print("\n--- 3. Analyst workload projection ---")
    n_flagged = int(((probs >= 0.05)).sum())
    project_analyst_workload(
        n_flagged_in_test=n_flagged,
        test_size=len(X_test),
        daily_txn_volume=2_000_000,
    )

    print("\n--- 4. Operational Payment Metrics ---")
    fpr_result = recall_at_fixed_fpr(y_test, probs, target_fpr=0.01)
    print(f"Recall @ 1.0% False Positive Rate (FPR): {fpr_result['recall_at_target_fpr']:.1%}")
    print(f"  Operating Threshold required: {fpr_result['threshold']:.4f} (Achieved FPR: {fpr_result['achieved_fpr']:.2%})")

    top_k_df = precision_at_k(y_test, probs, k_values=[50, 100, 120, 200])
    print("\nPrecision @ Top-K Review Capacity:")
    print(top_k_df.to_string(index=False))

    print("\n--- 5. Sensor / Telemetry Dropout Robustness ---")
    dropout_df = evaluate_feature_dropout(model, X_test, y_test)
    print(dropout_df.to_string(index=False))

    print("\n--- 6. Probability Calibration Quality ---")
    calib = evaluate_calibration(y_test, probs)
    print(f"Brier Score:                  {calib['brier_score']:.6f}  (closer to 0 is better)")
    print(f"Expected Calibration Error:   {calib['expected_calibration_error']:.4f}  (lower is better)")
