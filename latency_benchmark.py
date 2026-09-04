"""
Latency benchmark for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Measures how long a single prediction takes, to answer a real production
question: could this run in-line before a transaction completes, or only
as a post-hoc batch job?
"""

import os
import time
import numpy as np
import pandas as pd

from fraud_classifier import (
    generate_synthetic_data,
    engineer_features,
    train_model,
    FEATURES,
    load_artifacts,
)
from risk_pipeline import get_pipeline


def benchmark_single_prediction(model, X_test, n_runs=1000):
    """
    Times model.predict_proba on a single row, repeated n_runs times, to get
    a stable estimate (a single call is too noisy to trust on its own).
    """
    sample_row = X_test.iloc[[0]]

    # Warm-up call — first call often includes one-time overhead (JIT, caching)
    # that wouldn't reflect steady-state production latency.
    _ = model.predict_proba(sample_row)

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict_proba(sample_row)
        timings.append(time.perf_counter() - start)

    timings = np.array(timings) * 1000  # convert to milliseconds
    return {
        "mean_ms": timings.mean(),
        "median_ms": np.median(timings),
        "p95_ms": np.percentile(timings, 95),
        "p99_ms": np.percentile(timings, 99),
        "max_ms": timings.max(),
    }


def benchmark_batch_prediction(model, X_test, batch_size=1000):
    """
    Times scoring a realistic batch, to show throughput alongside single-row latency.
    """
    batch = X_test.iloc[:batch_size] if len(X_test) >= batch_size else X_test

    start = time.perf_counter()
    _ = model.predict_proba(batch)
    elapsed = time.perf_counter() - start

    return {
        "batch_size": len(batch),
        "total_ms": elapsed * 1000,
        "per_txn_ms": (elapsed * 1000) / len(batch),
        "txns_per_second": len(batch) / elapsed,
    }


def benchmark_pipeline_scoring(pipeline, txn_dict, n_runs=500, label=""):
    """
    Times the full RiskPipeline.score_single (profiling + inference + decision + safety cap + audit).
    """
    # Warm-up
    _ = pipeline.score_single(txn_dict, record_audit=False)

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = pipeline.score_single(txn_dict, record_audit=False)
        timings.append(time.perf_counter() - start)

    timings = np.array(timings) * 1000
    return {
        "label": label,
        "mean_ms": timings.mean(),
        "median_ms": np.median(timings),
        "p95_ms": timings.mean() if len(timings) < 20 else np.percentile(timings, 95),
        "p99_ms": timings.mean() if len(timings) < 20 else np.percentile(timings, 99),
        "max_ms": timings.max(),
    }


if __name__ == "__main__":
    if os.path.exists("artifacts/model.json") and os.path.exists("artifacts/user_baselines.json"):
        print("Loading pre-trained artifacts from 'artifacts/' (sub-second cold start)...")
        t0 = time.perf_counter()
        pipeline = get_pipeline()
        load_time = (time.perf_counter() - t0) * 1000
        print(f"Artifacts loaded in {load_time:.1f} ms.")
        model = pipeline.model
        df = generate_synthetic_data(n_txns=1000)
        df = pipeline.profiler.transform(df)
        X_test = df[FEATURES]
    else:
        print("No artifacts found; training model...")
        df = engineer_features(generate_synthetic_data())
        model, X_test, y_test = train_model(df)
        pipeline = get_pipeline()

    print("\nBenchmarking raw XGBoost single-transaction scoring (1000 runs)...")
    single = benchmark_single_prediction(model, X_test)

    print(f"  Mean:   {single['mean_ms']:.3f} ms")
    print(f"  Median: {single['median_ms']:.3f} ms")
    print(f"  P95:    {single['p95_ms']:.3f} ms")
    print(f"  P99:    {single['p99_ms']:.3f} ms")
    print(f"  Max:    {single['max_ms']:.3f} ms")

    print("\nBenchmarking batch scoring (1000 transactions)...")
    batch = benchmark_batch_prediction(model, X_test)
    print(f"  Total:        {batch['total_ms']:.2f} ms for {batch['batch_size']} transactions")
    print(f"  Per-txn:      {batch['per_txn_ms']:.4f} ms")
    print(f"  Throughput:   {batch['txns_per_second']:,.0f} transactions/second")

    print("\nBenchmarking full RiskPipeline.score_single end-to-end:")
    # Low risk (Allow path — no SHAP needed)
    low_risk_txn = {
        "user_id": 42,
        "amount": 250.0,
        "time_since_last_txn_min": 25.0,
        "txn_velocity_10min": 1,
        "device_change": 0,
        "geo_dist_from_usual_km": 2.0,
        "login_burst_count": 0,
        "ip_risk_score": 0.05,
        "hour_of_day": 14,
    }
    low_pipe = benchmark_pipeline_scoring(pipeline, low_risk_txn, n_runs=500, label="Low-risk (Allow)")
    print(f"  [Allow Path] Mean: {low_pipe['mean_ms']:.3f} ms | P95: {low_pipe['p95_ms']:.3f} ms | P99: {low_pipe['p99_ms']:.3f} ms")

    # High risk (Block path — includes on-demand SHAP generation)
    high_risk_txn = {
        "user_id": 42,
        "amount": 15000.0,
        "time_since_last_txn_min": 0.5,
        "txn_velocity_10min": 5,
        "device_change": 1,
        "geo_dist_from_usual_km": 450.0,
        "login_burst_count": 4,
        "ip_risk_score": 0.98,
        "hour_of_day": 3,
    }
    high_pipe = benchmark_pipeline_scoring(pipeline, high_risk_txn, n_runs=100, label="High-risk (Block + SHAP)")
    print(f"  [Flagged Path + SHAP] Mean: {high_pipe['mean_ms']:.3f} ms | P95: {high_pipe['p95_ms']:.3f} ms | P99: {high_pipe['p99_ms']:.3f} ms")

    print(f"\n{'='*60}")
    if single["p99_ms"] < 50:
        print(f">>> P99 raw model latency of {single['p99_ms']:.1f}ms (and {low_pipe['p99_ms']:.1f}ms full pipeline)")
        print(">>> is well within the typical <100-200ms budget for in-line scoring during")
        print(">>> payment authorization — this runs BEFORE a transaction completes.")
    else:
        print(f">>> P99 latency of {single['p99_ms']:.1f}ms may require optimization.")
