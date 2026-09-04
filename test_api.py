"""
FastAPI HTTP Endpoint Integration Tests.
Razorpay Buildathon — AI Risk Manager track.

Tests real HTTP endpoints using starlette/fastapi TestClient:
- Root health check
- Low-risk scoring (allow)
- High-risk scoring (block + TreeSHAP top factors)
- Dynamic UserProfiler z-score calculation when omitted
- Pydantic schema validation errors (422)
- Auto-block safety cap bursting / downgrade to review
- Append-only audit trail logging verification

Run with: pytest test_api.py -v
"""

import json
import os
import pytest
from fastapi.testclient import TestClient

from api import app, _pipeline
from decision_layer import SafetyCapManager


@pytest.fixture
def client():
    return TestClient(app)


class TestAPIEndpoints:

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Fraud-spike detector"
        assert data["status"] == "ready"
        assert "/score" in data["endpoints"]

    def test_score_low_risk_transaction_allowed(self, client):
        payload = {
            "user_id": 1,
            "amount": 120.0,
            "time_since_last_txn_min": 60.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 1.5,
            "login_burst_count": 0,
            "ip_risk_score": 0.02,
            "hour_of_day": 14,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "allow"
        assert data["risk_score"] < 0.05
        assert data["capped_by_safety_limit"] is False
        assert data["top_factors"] is None  # no SHAP overhead on allowed txns

    def test_score_high_risk_transaction_blocked_with_shap(self, client):
        # Temporarily ensure capacity on safety cap
        _pipeline.safety_cap.reset()

        payload = {
            "user_id": 99,
            "amount": 15000.0,
            "time_since_last_txn_min": 0.2,
            "txn_velocity_10min": 6,
            "device_change": 1,
            "geo_dist_from_usual_km": 480.0,
            "login_burst_count": 5,
            "ip_risk_score": 0.99,
            "hour_of_day": 3,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "block"
        assert data["risk_score"] >= 0.80
        assert data["capped_by_safety_limit"] is False
        assert data["top_factors"] is not None
        assert len(data["top_factors"]) == 3
        for factor in data["top_factors"]:
            assert "feature" in factor
            assert "value" in factor
            assert "contribution" in factor
            assert factor["direction"] in ("increased", "decreased")

    def test_score_omitted_amount_zscore_calculated_dynamically(self, client):
        payload = {
            "user_id": 42,
            "amount": 250.0,
            "time_since_last_txn_min": 30.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.1,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "action" in data

    def test_score_validation_error_ip_risk_score_out_of_bounds(self, client):
        payload = {
            "amount": 100.0,
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 1.8,  # invalid: > 1.0
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_score_validation_error_hour_out_of_bounds(self, client):
        payload = {
            "amount": 100.0,
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.2,
            "hour_of_day": 28,  # invalid: > 23
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_score_validation_error_missing_required_amount(self, client):
        payload = {
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.2,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_negative_amount_rejected_422(self, client):
        payload = {
            "amount": -50.0,  # invalid: <= 0
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.2,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_zero_amount_rejected_422(self, client):
        payload = {
            "amount": 0.0,  # invalid: must be > 0
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.2,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_negative_velocity_rejected_422(self, client):
        payload = {
            "amount": 100.0,
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": -2,  # invalid: < 0
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.2,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_negative_geo_distance_rejected_422(self, client):
        payload = {
            "amount": 100.0,
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": -15.0,  # invalid: < 0
            "login_burst_count": 0,
            "ip_risk_score": 0.2,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_negative_time_since_last_txn_rejected_422(self, client):
        payload = {
            "amount": 100.0,
            "time_since_last_txn_min": -5.0,  # invalid: < 0
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.2,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422

    def test_negative_login_burst_rejected_422(self, client):
        payload = {
            "amount": 100.0,
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": -1,  # invalid: < 0
            "ip_risk_score": 0.2,
            "hour_of_day": 12,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422


class TestAPISafetyCapAndAudit:

    def test_safety_cap_downgrades_excess_blocks(self, client):
        # Configure a local cap of 2 blocks for testing bursting
        orig_cap = _pipeline.safety_cap
        test_cap = SafetyCapManager(max_blocks_per_hour=2)
        _pipeline.safety_cap = test_cap

        try:
            high_risk_payload = {
                "user_id": 999,
                "amount": 20000.0,
                "amount_zscore": 6.0,
                "time_since_last_txn_min": 0.1,
                "txn_velocity_10min": 7,
                "device_change": 1,
                "geo_dist_from_usual_km": 500.0,
                "login_burst_count": 6,
                "ip_risk_score": 0.99,
                "hour_of_day": 2,
            }

            # Calls 1 & 2: allowed to block
            res1 = client.post("/score", json=high_risk_payload).json()
            assert res1["action"] == "block"
            assert res1["capped_by_safety_limit"] is False

            res2 = client.post("/score", json=high_risk_payload).json()
            assert res2["action"] == "block"
            assert res2["capped_by_safety_limit"] is False

            # Call 3: exceeds cap of 2, must be downgraded to review
            res3 = client.post("/score", json=high_risk_payload).json()
            assert res3["action"] == "review"
            assert res3["capped_by_safety_limit"] is True
        finally:
            _pipeline.safety_cap = orig_cap

    def test_audit_trail_recorded_on_score(self, client):
        audit_file = "audit_trail.jsonl"
        initial_count = 0
        if os.path.exists(audit_file):
            with open(audit_file, "r") as f:
                initial_count = len([line for line in f if line.strip()])

        payload = {
            "user_id": 77,
            "amount": 333.0,
            "time_since_last_txn_min": 15.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 10.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.05,
            "hour_of_day": 15,
        }
        res = client.post("/score", json=payload)
        assert res.status_code == 200

        assert os.path.exists(audit_file)
        with open(audit_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == initial_count + 1
        latest_record = json.loads(lines[-1])
        assert "timestamp" in latest_record
        assert "transaction_id" in latest_record
        assert "risk_score" in latest_record
        assert latest_record["action"] == res.json()["action"]


class TestFailSafeAndOperationalEndpoints:

    def test_shap_failure_fail_safe(self, client, monkeypatch):
        def mock_explainer_error(row):
            raise RuntimeError("Simulated SHAP timeout / computation error")

        monkeypatch.setattr(_pipeline, "explainer", mock_explainer_error)

        payload = {
            "user_id": 99,
            "amount": 15000.0,
            "time_since_last_txn_min": 0.2,
            "txn_velocity_10min": 6,
            "device_change": 1,
            "geo_dist_from_usual_km": 480.0,
            "login_burst_count": 5,
            "ip_risk_score": 0.99,
            "hour_of_day": 3,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Decision must remain unchanged (high risk -> block/review)
        assert data["action"] in ("block", "review")
        assert data["risk_score"] >= 0.80
        assert data["explanation_degraded"] is True
        assert data["top_factors"] is not None
        assert data["top_factors"][0]["direction"] == "degraded"

    def test_audit_logger_failure_fail_safe(self, client, monkeypatch):
        def mock_log_entry_error(entry):
            raise IOError("Simulated disk full / permissions write error")

        monkeypatch.setattr(_pipeline.audit_logger, "log_entry", mock_log_entry_error)

        payload = {
            "user_id": 1,
            "amount": 120.0,
            "time_since_last_txn_min": 60.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 1.5,
            "login_burst_count": 0,
            "ip_risk_score": 0.02,
            "hour_of_day": 14,
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "allow"

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["service"] == "Fraud-spike detector API"
        assert "uptime_seconds" in data
        assert data["model_loaded"] is True
        assert data["profiler_loaded"] is True
        assert "artifacts_on_disk" in data
        assert data["model_version"] == "1.0.0-xgb"
        assert "timestamp" in data

    def test_metrics_endpoint(self, client):
        payload = {
            "user_id": 10,
            "amount": 200.0,
            "time_since_last_txn_min": 10.0,
            "txn_velocity_10min": 1,
            "device_change": 0,
            "geo_dist_from_usual_km": 5.0,
            "login_burst_count": 0,
            "ip_risk_score": 0.05,
            "hour_of_day": 12,
        }
        client.post("/score", json=payload)

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] >= 1
        assert "uptime_seconds" in data
        assert "decisions" in data
        assert "allow" in data["decisions"]
        assert "safety_cap" in data
        assert data["safety_cap"]["max_blocks_per_hour"] == 50
        assert "remaining_block_capacity" in data["safety_cap"]
        assert "reliability" in data
        assert "degraded_explanations" in data["reliability"]
        assert "audit_failures" in data["reliability"]

    def test_safety_cap_reset_endpoint(self, client):
        response = client.post("/safety-cap/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"
        assert data["current_hourly_blocks"] == 0
        assert data["remaining_block_capacity"] == 50
