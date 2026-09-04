"""
Latency benchmark for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Measures how long a single prediction takes, to answer a real production
question: could this run in-line before a transaction completes, or only
as a post-hoc batch job?
"""

import time
import numpy as np
import pandas as pd

from fraud_classifier import generate_synthetic_data, engineer_features, train_model, FEATURES


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


if __name__ == "__main__":
    print("Training model...")
    df = engineer_features(generate_synthetic_data())
    model, X_test, y_test = train_model(df)

    print("\nBenchmarking single-transaction scoring (1000 runs)...")
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

    print(f"\n{'='*60}")
    if single["p99_ms"] < 50:
        print(f">>> P99 latency of {single['p99_ms']:.1f}ms is well within the typical")
        print(">>> <100-200ms budget for in-line scoring during a payment authorization")
        print(">>> flow — this could run BEFORE a transaction completes, not just")
        print(">>> flag it after the fact.")
    else:
        print(f">>> P99 latency of {single['p99_ms']:.1f}ms may be too slow for in-line")
        print(">>> scoring during checkout and would need optimization (e.g. a smaller")
        print(">>> model, feature caching) before real-time deployment.")
    print("Note: this excludes SHAP explanation time, which is only computed for")
    print("flagged transactions (review/block), not every transaction — see")
    print("shap_explainer.py and decision_layer.py for that separation.")
