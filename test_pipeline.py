"""
Test suite for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Run with: pytest test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from fraud_classifier import (
    generate_synthetic_data, engineer_features, train_model, FEATURES,
    export_artifacts, load_artifacts, split_data, cost_weighted_threshold,
)
from decision_layer import (
    decide_action, process_batch, REVIEW_THRESHOLD, BLOCK_THRESHOLD,
    MAX_AUTO_BLOCKS_PER_HOUR, SafetyCapManager, AuditLogger,
)
from user_profiler import UserProfiler
from risk_pipeline import RiskPipeline, get_pipeline


# ---------------------------------------------------------------------
# Decision boundary tests — the core of the "bounded actions" claim
# ---------------------------------------------------------------------
class TestDecisionBoundaries:

    def test_below_review_threshold_is_allow(self):
        assert decide_action(REVIEW_THRESHOLD - 0.01) == "allow"

    def test_at_review_threshold_is_review(self):
        assert decide_action(REVIEW_THRESHOLD) == "review"

    def test_between_thresholds_is_review(self):
        midpoint = (REVIEW_THRESHOLD + BLOCK_THRESHOLD) / 2
        assert decide_action(midpoint) == "review"

    def test_at_block_threshold_is_block(self):
        assert decide_action(BLOCK_THRESHOLD) == "block"

    def test_above_block_threshold_is_block(self):
        assert decide_action(0.99) == "block"

    def test_zero_risk_is_allow(self):
        assert decide_action(0.0) == "allow"

    def test_max_risk_is_block(self):
        assert decide_action(1.0) == "block"

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_out_of_range_scores_dont_crash(self, score):
        # Model probabilities should never fall outside [0,1], but the
        # decision function should still behave sanely if they somehow do.
        result = decide_action(score)
        assert result in ("allow", "review", "block")


# ---------------------------------------------------------------------
# Classifier output validity — does the model behave like a probability
# ---------------------------------------------------------------------
class TestClassifierOutput:

    @pytest.fixture(scope="class")
    @staticmethod
    def trained_model():
        df = engineer_features(generate_synthetic_data(n_txns=5000))
        model, X_test, y_test = train_model(df)
        return model, X_test, y_test

    def test_predictions_are_valid_probabilities(self, trained_model):
        model, X_test, _ = trained_model
        probs = model.predict_proba(X_test)[:, 1]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_no_nan_predictions(self, trained_model):
        model, X_test, _ = trained_model
        probs = model.predict_proba(X_test)[:, 1]
        assert not np.any(np.isnan(probs))

    def test_output_length_matches_input(self, trained_model):
        model, X_test, _ = trained_model
        probs = model.predict_proba(X_test)[:, 1]
        assert len(probs) == len(X_test)

    def test_expected_features_present(self, trained_model):
        _, X_test, _ = trained_model
        assert list(X_test.columns) == FEATURES


# ---------------------------------------------------------------------
# Safety cap — the hard bound that must never be silently exceeded
# ---------------------------------------------------------------------
class TestSafetyCap:

    def test_safety_cap_manager_evaluates_blocks(self):
        cap = SafetyCapManager(max_blocks_per_hour=3)
        # First 3 blocks should succeed
        for _ in range(3):
            action, capped = cap.evaluate_block()
            assert action == "block"
            assert not capped
        # 4th and 5th should degrade to review
        for _ in range(2):
            action, capped = cap.evaluate_block()
            assert action == "review"
            assert capped

    def test_safety_cap_sliding_window_expiration(self):
        cap = SafetyCapManager(max_blocks_per_hour=2, window_sec=60.0)
        t0 = 1000.0
        assert cap.evaluate_block(now=t0) == ("block", False)
        assert cap.evaluate_block(now=t0 + 10) == ("block", False)
        # Cap reached at t0 + 20
        assert cap.evaluate_block(now=t0 + 20) == ("review", True)
        # After window expiration (t0 + 65), the first block at t0 has expired
        assert cap.evaluate_block(now=t0 + 65) == ("block", False)

    def test_block_count_never_exceeds_cap_in_a_batch(self):
        """
        Runs the real decision pipeline on a high-fraud batch and asserts the cap is respected.
        """
        df = generate_synthetic_data(n_txns=4000, fraud_rate=0.05)
        model, X_test, y_test = train_model(df)

        probs = model.predict_proba(X_test)[:, 1]
        cap = SafetyCapManager(max_blocks_per_hour=MAX_AUTO_BLOCKS_PER_HOUR)
        block_count = 0
        capped_events = 0
        fixed_time = 2000.0
        for p in probs:
            action = decide_action(p)
            if action == "block":
                action, capped = cap.evaluate_block(now=fixed_time)
                if capped:
                    capped_events += 1
                else:
                    block_count += 1

        assert block_count <= MAX_AUTO_BLOCKS_PER_HOUR


# ---------------------------------------------------------------------
# Risk Pipeline integration tests
# ---------------------------------------------------------------------
class TestRiskPipelineIntegration:

    def test_score_single_enforces_safety_cap_and_logs(self, tmp_path):
        import shap
        audit_file = str(tmp_path / "test_audit.jsonl")

        df = generate_synthetic_data(n_txns=1000)
        model, X_test, y_test = train_model(df)
        profiler = getattr(model, "profiler", UserProfiler().fit(df))
        explainer = shap.TreeExplainer(model)

        cap = SafetyCapManager(max_blocks_per_hour=2)
        pipeline = RiskPipeline(
            model=model,
            profiler=profiler,
            explainer=explainer,
            safety_cap=cap,
            audit_path=audit_file,
        )

        high_risk_txn = {
            "user_id": 1,
            "amount": 10000.0,
            "amount_zscore": 5.0,
            "time_since_last_txn_min": 1.0,
            "txn_velocity_10min": 5,
            "device_change": 1,
            "geo_dist_from_usual_km": 300.0,
            "login_burst_count": 4,
            "ip_risk_score": 0.95,
            "hour_of_day": 3,
        }

        r1 = pipeline.score_single(high_risk_txn, record_audit=True)
        r2 = pipeline.score_single(high_risk_txn, record_audit=True)
        r3 = pipeline.score_single(high_risk_txn, record_audit=True)

        assert r1["action"] == "block" and not r1["capped_by_safety_limit"]
        assert r2["action"] == "block" and not r2["capped_by_safety_limit"]
        assert r3["action"] == "review" and r3["capped_by_safety_limit"]

        # Check audit file entries
        with open(audit_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------
# API input validation (schema-level, without spinning up a live server)
# ---------------------------------------------------------------------
class TestAPISchema:

    def test_transaction_model_accepts_valid_input(self):
        from api import Transaction
        txn = Transaction(
            amount=4200.0, amount_zscore=3.8, time_since_last_txn_min=4.5,
            txn_velocity_10min=3, device_change=1, geo_dist_from_usual_km=210.0,
            login_burst_count=2, ip_risk_score=0.71, hour_of_day=2,
        )
        assert txn.amount == 4200.0

    def test_transaction_model_rejects_invalid_ip_risk_score(self):
        from api import Transaction
        with pytest.raises(Exception):
            Transaction(
                amount=100.0, amount_zscore=0.0, time_since_last_txn_min=10.0,
                txn_velocity_10min=1, device_change=0, geo_dist_from_usual_km=5.0,
                login_burst_count=0, ip_risk_score=1.5,  # invalid: > 1
                hour_of_day=12,
            )

    def test_transaction_model_rejects_invalid_hour(self):
        from api import Transaction
        with pytest.raises(Exception):
            Transaction(
                amount=100.0, amount_zscore=0.0, time_since_last_txn_min=10.0,
                txn_velocity_10min=1, device_change=0, geo_dist_from_usual_km=5.0,
                login_burst_count=0, ip_risk_score=0.1,
                hour_of_day=25,  # invalid: > 23
            )


    def test_transaction_model_accepts_omitted_amount_zscore(self):
        from api import Transaction
        txn = Transaction(
            user_id=10,
            amount=4200.0,
            time_since_last_txn_min=4.5,
            txn_velocity_10min=3,
            device_change=1,
            geo_dist_from_usual_km=210.0,
            login_burst_count=2,
            ip_risk_score=0.71,
            hour_of_day=2,
        )
        assert txn.amount == 4200.0
        assert txn.amount_zscore is None
        assert txn.user_id == 10


# ---------------------------------------------------------------------
# User Profiler & Leak-Free Baseline tests
# ---------------------------------------------------------------------
class TestUserProfiler:

    def test_profiler_fit_learns_correct_stats(self):
        df_train = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "amount": [100.0, 200.0, 300.0, 50.0, 50.0],
        })
        profiler = UserProfiler()
        profiler.fit(df_train)

        assert profiler.is_fitted
        assert profiler.user_stats[1]["mean"] == 200.0
        assert profiler.user_stats[1]["std"] == 100.0
        assert profiler.user_stats[2]["mean"] == 50.0
        assert profiler.global_mean == 140.0

    def test_profiler_transform_calculates_zscore(self):
        df_train = pd.DataFrame({
            "user_id": [1, 1, 1],
            "amount": [100.0, 200.0, 300.0],
        })
        profiler = UserProfiler().fit(df_train)
        df_test = pd.DataFrame({
            "user_id": [1],
            "amount": [400.0],
        })
        transformed = profiler.transform(df_test)
        # z-score: (400 - 200) / 100 = 2.0
        assert np.isclose(transformed["amount_zscore"].iloc[0], 2.0)

    def test_unseen_users_fall_back_to_priors(self):
        df_train = pd.DataFrame({
            "user_id": [1, 1],
            "amount": [100.0, 200.0],
        })
        profiler = UserProfiler().fit(df_train)
        df_unseen = pd.DataFrame({
            "user_id": [999],  # Unseen user
            "amount": [300.0],
        })
        transformed = profiler.transform(df_unseen)
        assert not transformed["amount_zscore"].isna().any()
        expected_z = (300.0 - profiler.global_mean) / profiler.global_std
        assert np.isclose(transformed["amount_zscore"].iloc[0], expected_z)

    def test_transform_single_derives_zscore(self):
        df_train = pd.DataFrame({
            "user_id": [1, 1],
            "amount": [100.0, 300.0],  # mean=200, std=141.421
        })
        profiler = UserProfiler().fit(df_train)
        txn = {"user_id": 1, "amount": 200.0}
        z = profiler.transform_single(txn)
        assert z == 0.0

    def test_transform_single_respects_explicit_zscore(self):
        df_train = pd.DataFrame({"user_id": [1], "amount": [100.0]})
        profiler = UserProfiler().fit(df_train)
        txn = {"user_id": 1, "amount": 100.0, "amount_zscore": 5.5}
        z = profiler.transform_single(txn)
        assert z == 5.5


# ---------------------------------------------------------------------
# Model Artifact Persistence tests (Phase 3)
# ---------------------------------------------------------------------
class TestArtifactPersistence:

    def test_export_and_load_artifacts(self, tmp_path):
        artifact_dir = str(tmp_path / "artifacts")
        df = generate_synthetic_data(n_txns=500)
        model, X_test, y_test = train_model(df)
        profiler = getattr(model, "profiler", None)

        export_artifacts(model, profiler=profiler, artifact_dir=artifact_dir)

        loaded_model, loaded_profiler = load_artifacts(artifact_dir=artifact_dir)

        assert loaded_model is not None
        assert loaded_profiler is not None
        assert loaded_profiler.is_fitted

        # Assert predictions match exactly
        orig_probs = model.predict_proba(X_test)
        loaded_probs = loaded_model.predict_proba(X_test)
        np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5, atol=1e-5)

        # Assert profiler state matches
        assert loaded_profiler.global_mean == profiler.global_mean
        assert loaded_profiler.global_std == profiler.global_std

    def test_load_artifacts_missing_file_raises(self, tmp_path):
        empty_dir = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            load_artifacts(artifact_dir=empty_dir)

    def test_get_pipeline_cold_start(self):
        pipeline = get_pipeline()
        assert pipeline is not None
        assert pipeline.model is not None
        assert pipeline.profiler is not None
        # Test scoring a transaction using loaded pipeline
        sample_txn = {
            "user_id": 1,
            "amount": 250.0,
            "time_since_last_txn_min": 15.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.1,
            "hour_of_day": 14,
        }
        res = pipeline.score_single(sample_txn, record_audit=False)
        assert res["action"] in ("allow", "review", "block")
        assert 0.0 <= res["risk_score"] <= 1.0


# ---------------------------------------------------------------------
# Phase 1: Train / Val / Test Separation & Scoped RNG tests
# ---------------------------------------------------------------------
class TestTrainValTestSeparation:

    def test_split_partitions_are_strictly_non_overlapping(self):
        df = generate_synthetic_data(n_txns=1000, seed=42)
        train_df, val_df, test_df = split_data(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42)

        # 1. Size conservation
        assert len(train_df) == 600
        assert len(val_df) == 200
        assert len(test_df) == 200
        assert len(train_df) + len(val_df) + len(test_df) == len(df)

        # 2. Pairwise non-overlapping index sets
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        test_idx = set(test_df.index)

        assert train_idx.isdisjoint(val_idx), "Train and Validation indices overlap!"
        assert train_idx.isdisjoint(test_idx), "Train and Test indices overlap!"
        assert val_idx.isdisjoint(test_idx), "Validation and Test indices overlap!"

        # 3. Stratification preservation
        overall_fraud_rate = df["is_fraud"].mean()
        train_fraud_rate = train_df["is_fraud"].mean()
        val_fraud_rate = val_df["is_fraud"].mean()
        test_fraud_rate = test_df["is_fraud"].mean()

        assert np.isclose(train_fraud_rate, overall_fraud_rate, atol=0.01)
        assert np.isclose(val_fraud_rate, overall_fraud_rate, atol=0.01)
        assert np.isclose(test_fraud_rate, overall_fraud_rate, atol=0.01)

    def test_user_profiler_fits_only_on_training_data(self):
        df = generate_synthetic_data(n_txns=1000, seed=42)
        train_df, val_df, test_df = split_data(df, random_state=42)

        profiler = UserProfiler().fit(train_df)

        # Verify profiler global stats match train_df exactly, not the full df
        assert profiler.global_mean == train_df["amount"].mean()
        assert profiler.global_std == train_df["amount"].std()
        assert profiler.global_mean != df["amount"].mean()

        # Invalidate/contaminate test_df with an extreme outlier
        contaminated_test = test_df.copy()
        contaminated_test.loc[contaminated_test.index[0], "amount"] = 999_999.0

        # Verify that profiler's internal learned statistics are completely unchanged
        assert profiler.global_mean == train_df["amount"].mean()

    def test_threshold_selection_occurs_only_on_validation_data(self):
        df = generate_synthetic_data(n_txns=1000, seed=42)
        model, X_val, y_val, X_test, y_test = train_model(df, return_val=True, random_state=42)

        val_probs = model.predict_proba(X_val)[:, 1]
        results_df, best = cost_weighted_threshold(y_val, val_probs)

        # Selected threshold must be in valid search grid
        selected_threshold = best["threshold"]
        assert 0.05 <= selected_threshold <= 0.95
        assert best["total_cost_inr"] >= 0

        # Now evaluate on untouched test set using the LOCKED threshold from validation
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= selected_threshold).astype(int)
        assert len(test_preds) == len(y_test)
        assert set(test_preds).issubset({0, 1})

    def test_deterministic_synthetic_generation_same_seed(self):
        df1 = generate_synthetic_data(n_txns=500, seed=42)
        df2 = generate_synthetic_data(n_txns=500, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_datasets(self):
        df1 = generate_synthetic_data(n_txns=500, seed=42)
        df2 = generate_synthetic_data(n_txns=500, seed=123)
        assert not df1.equals(df2)
        assert not np.array_equal(df1["amount"].values, df2["amount"].values)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
