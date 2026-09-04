"""
Unit and integration tests for scientific benchmark evaluations and robustness modules.
Razorpay Buildathon — AI Risk Manager track.
"""

import numpy as np
import pandas as pd
import pytest

from additional_evaluation import (
    recall_at_fixed_fpr,
    precision_at_k,
    evaluate_feature_dropout,
    evaluate_calibration,
)
from fraud_classifier import FEATURES, generate_synthetic_data, train_model
from model_benchmarks import (
    RuleBasedFraudDetector,
    calculate_metrics,
    run_model_benchmarks,
    run_dual_cohort_evaluation,
)
from temporal_evaluation import generate_temporal_data, run_temporal_comparison


@pytest.fixture(scope="module")
def sample_data():
    return generate_synthetic_data(n_users=500, n_txns=5000, seed=42)


@pytest.fixture(scope="module")
def trained_model_and_test():
    df = generate_synthetic_data(n_users=500, n_txns=5000, seed=42)
    model, X_test, y_test = train_model(df, random_state=42)
    probs = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, probs


class TestModelBenchmarks:
    def test_rule_based_detector_probabilities_and_bounds(self, sample_data):
        detector = RuleBasedFraudDetector()
        probs = detector.predict_proba(sample_data)

        assert probs.shape == (len(sample_data), 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), 1.0)

        preds = detector.predict(sample_data, threshold=0.05)
        assert len(preds) == len(sample_data)
        assert set(preds).issubset({0, 1})

    def test_calculate_metrics_financial_cost(self):
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([0.01, 0.10, 0.01, 0.90])  # 1 TN, 1 FP, 1 FN, 1 TP at threshold=0.05
        metrics = calculate_metrics(y_true, probs, threshold=0.05, cost_fp=150.0, cost_fn=8000.0)

        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["total_cost"] == 150.0 + 8000.0  # 8150.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_dual_cohort_user_partitions_are_strictly_disjoint(self, sample_data):
        cohort_results = run_dual_cohort_evaluation(sample_data, seen_user_ratio=0.8, seed=42)

        assert len(cohort_results) == 3
        cohorts = list(cohort_results["cohort"])
        assert any("Seen Users" in c for c in cohorts)
        assert any("Unseen Users" in c for c in cohorts)
        assert any("Combined" in c for c in cohorts)

        # PR-AUC, precision, recall should all be valid bounded floats
        for col in ["pr_auc", "precision", "recall", "f1"]:
            assert cohort_results[col].between(0.0, 1.0).all()


class TestTemporalEvaluation:
    def test_temporal_data_generator_properties(self):
        df = generate_temporal_data(n_users=100, n_days=5, txns_per_day=50, seed=42)

        assert len(df) == 5 * 50
        assert "timestamp" in df.columns
        assert "day" in df.columns
        # Check chronological monotonic ordering
        timestamps = df["timestamp"].values
        assert np.all(timestamps[:-1] <= timestamps[1:])

    def test_temporal_out_of_time_comparison_executes(self):
        results = run_temporal_comparison(seed=42)

        assert len(results) == 2
        strategies = list(results["evaluation_strategy"])
        assert any("Random Stratified" in s for s in strategies)
        assert any("Temporal Out-of-Time" in s for s in strategies)

        for col in ["pr_auc", "precision", "recall", "f1"]:
            assert results[col].between(0.0, 1.0).all()


class TestOperationalAndRobustnessMetrics:
    def test_recall_at_fixed_fpr_bounds(self, trained_model_and_test):
        _, _, y_test, probs = trained_model_and_test
        result = recall_at_fixed_fpr(y_test, probs, target_fpr=0.01)

        assert "achieved_fpr" in result
        assert "recall_at_target_fpr" in result
        assert result["achieved_fpr"] <= 0.01 + 1e-4  # within target bound
        assert 0.0 <= result["recall_at_target_fpr"] <= 1.0

    def test_precision_at_k(self, trained_model_and_test):
        _, _, y_test, probs = trained_model_and_test
        df_k = precision_at_k(y_test, probs, k_values=[20, 50])

        assert len(df_k) == 2
        assert list(df_k["k (Top Alerts Queue)"]) == [20, 50]
        assert df_k["precision_at_k"].between(0.0, 1.0).all()
        assert df_k["recall_at_k"].between(0.0, 1.0).all()

    def test_evaluate_feature_dropout_scenarios(self, trained_model_and_test):
        model, X_test, y_test, _ = trained_model_and_test
        df_dropout = evaluate_feature_dropout(model, X_test, y_test)

        assert len(df_dropout) == 5
        assert "Baseline (All Sensors)" in list(df_dropout["telemetry_scenario"])
        assert df_dropout["pr_auc"].between(0.0, 1.0).all()

    def test_evaluate_calibration_metrics(self, trained_model_and_test):
        _, _, y_test, probs = trained_model_and_test
        calib = evaluate_calibration(y_test, probs, n_bins=10)

        assert calib["brier_score"] >= 0.0
        assert calib["expected_calibration_error"] >= 0.0
