"""
Model benchmark comparison and user generalization analysis for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Evaluates:
  1. Standardized Model Baselines:
     - Heuristic Rule-Based Anomaly Detector (traditional gateway rules)
     - Logistic Regression (standardized linear baseline)
     - Random Forest (parallel bagging ensemble)
     - XGBoost Classifier (production gradient boosting model)
  2. Dual-Cohort User Generalization:
     - Seen User Cohort: users with established behavioral baselines in training
     - Unseen User Cohort: completely new users / cold-start accounts relying on population priors
"""

import time
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_classifier import (
    FEATURES,
    generate_synthetic_data,
    split_data,
)
from user_profiler import UserProfiler


# ---------------------------------------------------------------------
# 1. Heuristic Rule-Based Anomaly Detector (Industry Heuristic Engine)
# ---------------------------------------------------------------------
class RuleBasedFraudDetector:
    """
    Simulates a traditional payments rules engine.
    Uses boolean business logic over behavioral deviations and risk signals.
    """
    def __init__(
        self,
        geo_threshold_km: float = 100.0,
        login_burst_min: int = 2,
        velocity_min: int = 4,
        amount_z_min: float = 3.0,
        ip_risk_min: float = 0.70,
    ):
        self.geo_threshold_km = geo_threshold_km
        self.login_burst_min = login_burst_min
        self.velocity_min = velocity_min
        self.amount_z_min = amount_z_min
        self.ip_risk_min = ip_risk_min

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "RuleBasedFraudDetector":
        # Rules engines are expert-configured; no training required
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns pseudo-probabilities:
          - 0.95 for high-confidence rule match
          - 0.60 for moderate rule match
          - 0.01 for clean transactions
        """
        probs = np.full(len(X), 0.01)

        # Determine amount deviation signal (support both pre-profiled and raw DataFrames)
        if "amount_zscore" in X.columns:
            amt_z = X["amount_zscore"]
        elif "amount" in X.columns:
            mean_amt = float(X["amount"].mean())
            std_amt = float(X["amount"].std())
            std_amt = std_amt if std_amt > 1e-3 else 1.0
            amt_z = (X["amount"] - mean_amt) / std_amt
        else:
            amt_z = pd.Series(0.0, index=X.index)

        # Rule A (ATO Trigger): New device + substantial geo jump + login burst
        rule_ato = (
            (X["device_change"] == 1)
            & (X["geo_dist_from_usual_km"] >= self.geo_threshold_km)
            & (X["login_burst_count"] >= self.login_burst_min)
        )

        # Rule B (Velocity & Amount Spike): High velocity + extreme deviation in amount
        rule_velocity = (
            (X["txn_velocity_10min"] >= self.velocity_min)
            & (amt_z >= self.amount_z_min)
        )

        # Rule C (Threat IP + Rapid succession + Amount Spike)
        rule_ip = (
            (X["ip_risk_score"] >= self.ip_risk_min)
            & (X["time_since_last_txn_min"] <= 5.0)
            & (amt_z >= 2.0)
        )

        high_risk = rule_ato | rule_velocity | rule_ip
        moderate_risk = (
            (~high_risk)
            & ((X["device_change"] == 1) | (amt_z >= 2.5) | (X["ip_risk_score"] >= 0.75))
        )

        probs[moderate_risk] = 0.60
        probs[high_risk] = 0.95

        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: pd.DataFrame, threshold: float = 0.05) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------------
# 2. Benchmark Runner
# ---------------------------------------------------------------------
def calculate_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.05,
    cost_fp: float = 150.0,
    cost_fn: float = 8000.0,
) -> Dict[str, Any]:
    """Calculates PR-AUC, precision, recall, F1, and total financial cost."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    total_cost = fp * cost_fp + fn * cost_fn

    return {
        "pr_auc": average_precision_score(y_true, probs),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_cost": float(total_cost),
    }


def run_model_benchmarks(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Trains and evaluates 4 models on strictly split-safe Train / Val / Test partitions.
    """
    train_df, val_df, test_df = split_data(df, random_state=seed)

    # Fit UserProfiler strictly on training partition
    profiler = UserProfiler().fit(train_df)
    train_proc = profiler.transform(train_df)
    test_proc = profiler.transform(test_df)

    X_train = train_proc[FEATURES]
    y_train = train_proc["is_fraud"]
    X_test = test_proc[FEATURES]
    y_test = test_proc["is_fraud"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Heuristic Rules Engine": RuleBasedFraudDetector(),
        "Logistic Regression (Balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ]),
        "Random Forest (100 trees)": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "XGBoost Classifier (Production)": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=seed,
        ),
    }

    results = []
    print("\n" + "=" * 80)
    print("MODEL BENCHMARK COMPARISON (Evaluated on Test Set, n=8,000)")
    print("=" * 80)

    for name, model in models.items():
        # Fit model
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_duration = time.perf_counter() - fit_start

        # Measure batch inference and per-item latency
        score_start = time.perf_counter()
        probs = model.predict_proba(X_test)[:, 1]
        score_duration = time.perf_counter() - score_start
        per_txn_latency_ms = (score_duration / len(X_test)) * 1000

        metrics = calculate_metrics(y_test, probs, threshold=0.05)
        metrics["model"] = name
        metrics["fit_time_sec"] = round(fit_duration, 3)
        metrics["per_txn_latency_ms"] = round(per_txn_latency_ms, 4)
        results.append(metrics)

    res_df = pd.DataFrame(results)[[
        "model", "pr_auc", "precision", "recall", "f1",
        "false_positives", "false_negatives", "total_cost", "per_txn_latency_ms"
    ]]
    return res_df


# ---------------------------------------------------------------------
# 3. Dual-Cohort User Generalization Analysis (Seen vs. Unseen)
# ---------------------------------------------------------------------
def run_dual_cohort_evaluation(df: pd.DataFrame, seen_user_ratio: float = 0.8, seed: int = 42) -> pd.DataFrame:
    """
    Evaluates generalization across two distinct populations:
      - Seen Users (80%): have transaction history in the training set.
      - Unseen Users (20%): brand-new accounts / cold-start users with zero training history.
    """
    rng = np.random.default_rng(seed)
    all_users = np.array(sorted(df["user_id"].unique()))
    n_seen = int(len(all_users) * seen_user_ratio)

    # Permute and partition users
    shuffled_users = rng.permutation(all_users)
    seen_users = set(shuffled_users[:n_seen])
    unseen_users = set(shuffled_users[n_seen:])

    # Partition transactions
    seen_df = df[df["user_id"].isin(seen_users)].copy()
    unseen_test_df = df[df["user_id"].isin(unseen_users)].copy()

    # Split seen users into Train (75%) and Test-Seen (25%)
    # This corresponds to roughly 60% Train, 20% Test-Seen, 20% Test-Unseen of total
    train_seen_df, test_seen_df = train_test_split(
        seen_df, test_size=0.25, stratify=seen_df["is_fraud"], random_state=seed
    )

    # Fit UserProfiler STRICTLY on train_seen_df
    profiler = UserProfiler().fit(train_seen_df)

    train_proc = profiler.transform(train_seen_df)
    test_seen_proc = profiler.transform(test_seen_df)
    test_unseen_proc = profiler.transform(unseen_test_df)

    X_train, y_train = train_proc[FEATURES], train_proc["is_fraud"]
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

    # Evaluate on Seen Cohort
    probs_seen = model.predict_proba(test_seen_proc[FEATURES])[:, 1]
    m_seen = calculate_metrics(test_seen_proc["is_fraud"], probs_seen, threshold=0.05)
    m_seen["cohort"] = "Seen Users (Historical Baseline Active)"
    m_seen["n_transactions"] = len(test_seen_proc)
    m_seen["fraud_count"] = int(test_seen_proc["is_fraud"].sum())

    # Evaluate on Unseen Cohort (Cold Start)
    probs_unseen = model.predict_proba(test_unseen_proc[FEATURES])[:, 1]
    m_unseen = calculate_metrics(test_unseen_proc["is_fraud"], probs_unseen, threshold=0.05)
    m_unseen["cohort"] = "Unseen Users (Cold Start Prior Fallback)"
    m_unseen["n_transactions"] = len(test_unseen_proc)
    m_unseen["fraud_count"] = int(test_unseen_proc["is_fraud"].sum())

    # Combined evaluation
    combined_test = pd.concat([test_seen_proc, test_unseen_proc], ignore_index=True)
    probs_combined = np.concatenate([probs_seen, probs_unseen])
    m_combined = calculate_metrics(combined_test["is_fraud"], probs_combined, threshold=0.05)
    m_combined["cohort"] = "Combined Population (Blended)"
    m_combined["n_transactions"] = len(combined_test)
    m_combined["fraud_count"] = int(combined_test["is_fraud"].sum())

    cohort_df = pd.DataFrame([m_seen, m_unseen, m_combined])[[
        "cohort", "n_transactions", "fraud_count", "pr_auc", "precision", "recall", "f1",
        "false_positives", "false_negatives", "total_cost"
    ]]

    return cohort_df


# ---------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic transactions (n=40,000, n_users=2,000)...")
    data = generate_synthetic_data(n_txns=40000, seed=42)

    # 1. Model benchmarks
    bench_results = run_model_benchmarks(data, seed=42)
    print(bench_results.to_string(index=False))

    # 2. Dual-cohort generalization
    print("\n" + "=" * 80)
    print("DUAL-COHORT GENERALIZATION ANALYSIS (Seen vs. Unseen / Cold-Start Users)")
    print("=" * 80)
    cohort_results = run_dual_cohort_evaluation(data, seen_user_ratio=0.8, seed=42)
    print(cohort_results.to_string(index=False))

    seen_auc = cohort_results.loc[cohort_results["cohort"].str.startswith("Seen"), "pr_auc"].values[0]
    unseen_auc = cohort_results.loc[cohort_results["cohort"].str.startswith("Unseen"), "pr_auc"].values[0]
    auc_diff = seen_auc - unseen_auc
    print(f"\nCold-Start Generalization Penalty: {auc_diff:+.4f} PR-AUC")
    print("Interpretation:")
    print("  - Seen users benefit from personalized spending baselines (mean & std).")
    print("  - Unseen users fall back to global population priors, illustrating the realistic")
    print("    challenge of cold-start fraud detection where personalized baselines do not yet exist.")
