"""
Mobile device-signal fusion for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Account takeover shows up in HOW someone interacts with their phone, not
just WHAT they transact. This extends the transaction-only classifier with
simulated mobile behavioral signals and measures whether they add real
predictive value — not just bolted on for novelty.

Signals modeled (each grounded in a real ATO tell):
  - touch_pressure_variance   : erratic/unfamiliar handling of the device
  - typing_rhythm_deviation   : deviation from the user's own typing cadence
  - accel_stability_score     : device motion pattern during the transaction
                                 (a stolen/borrowed phone moves differently
                                 than one resting in a familiar hand)
  - orientation_changes       : screen rotations during the session
  - app_session_duration_sec  : time in-app before the transaction (ATO is
                                 often rushed — much shorter than a genuine
                                 user's normal browse-then-pay flow)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

from fraud_classifier import generate_synthetic_data, engineer_features, FEATURES as TXN_FEATURES
from user_profiler import UserProfiler

RNG = np.random.default_rng(7)

MOBILE_FEATURES = [
    "touch_pressure_variance", "typing_rhythm_deviation",
    "accel_stability_score", "orientation_changes", "app_session_duration_sec",
]
ALL_FEATURES = TXN_FEATURES + MOBILE_FEATURES


# ---------------------------------------------------------------------
# Add simulated device/behavioral signals on top of the transaction data
# ---------------------------------------------------------------------
def add_mobile_signals(df):
    df = df.copy()
    n = len(df)
    is_fraud = df["is_fraud"].values.astype(bool)

    # Legit sessions: familiar handling, steady rhythm, calm device, longer browse —
    # but with real-world noise, so some legit sessions still look a bit "off"
    touch_legit = RNG.normal(0.22, 0.14, n)
    typing_legit = RNG.normal(28, 18, n)
    accel_legit = RNG.normal(0.72, 0.16, n)          # higher = more stable
    orient_legit = RNG.poisson(0.5, n)
    duration_legit = RNG.gamma(shape=2.2, scale=22, size=n)

    # ATO sessions: unfamiliar device handling, rushed, jittery, brief session —
    # but overlapping with legit, since a calm, practiced attacker looks less
    # different, and a nervous legit user (bad signal, low battery, etc.) looks
    # more different than typical
    touch_fraud = RNG.normal(0.42, 0.20, n)
    typing_fraud = RNG.normal(48, 26, n)
    accel_fraud = RNG.normal(0.50, 0.20, n)
    orient_fraud = RNG.poisson(1.0, n)
    duration_fraud = RNG.gamma(shape=1.5, scale=14, size=n)

    df["touch_pressure_variance"] = np.where(is_fraud,
        np.clip(touch_fraud, 0, 1), np.clip(touch_legit, 0, 1))
    df["typing_rhythm_deviation"] = np.where(is_fraud,
        np.clip(typing_fraud, 0, None), np.clip(typing_legit, 0, None))
    df["accel_stability_score"] = np.where(is_fraud,
        np.clip(accel_fraud, 0, 1), np.clip(accel_legit, 0, 1))
    df["orientation_changes"] = np.where(is_fraud, orient_fraud, orient_legit)
    df["app_session_duration_sec"] = np.where(is_fraud,
        np.clip(duration_fraud, 1, None), np.clip(duration_legit, 1, None))

    return df


# ---------------------------------------------------------------------
# Train + evaluate on a given feature set
# ---------------------------------------------------------------------
def train_and_eval(train_df, test_df, feature_list, label):
    X_train = train_df[feature_list]
    y_train = train_df["is_fraud"]
    X_test = test_df[feature_list]
    y_test = test_df["is_fraud"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = {
        "label": label,
        "pr_auc": average_precision_score(y_test, probs),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
    }
    return model, results


# ---------------------------------------------------------------------
# Run the comparison
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating data with simulated mobile behavioral signals...")
    df = generate_synthetic_data()
    df = add_mobile_signals(df)

    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["is_fraud"], random_state=42
    )
    profiler = UserProfiler().fit(train_df)
    train_df = profiler.transform(train_df)
    test_df = profiler.transform(test_df)

    print("\nTraining baseline (transaction features only)...")
    _, baseline = train_and_eval(train_df, test_df, TXN_FEATURES, "Transaction-only (baseline)")

    print("Training fused model (transaction + mobile behavioral signals)...")
    fused_model, fused = train_and_eval(train_df, test_df, ALL_FEATURES, "Transaction + mobile signals")

    comparison = pd.DataFrame([baseline, fused]).set_index("label")
    print("\n" + "=" * 60)
    print("Comparison: does mobile signal fusion actually help?")
    print("=" * 60)
    print(comparison.round(4).to_string())

    delta_recall = fused["recall"] - baseline["recall"]
    delta_precision = fused["precision"] - baseline["precision"]
    print(f"\nRecall change:    {delta_recall:+.3f}")
    print(f"Precision change: {delta_precision:+.3f}")

    if delta_recall > 0.005 or delta_precision > 0.005:
        print("\n>>> Mobile signals provide a measurable improvement over transaction data alone.")
    else:
        print("\n>>> Mobile signals show negligible improvement here — honest result, "
              "possibly because the synthetic transaction features already separate "
              "classes well. Real-world gains would likely be larger where transaction "
              "signal alone is weaker (e.g. a stolen card used within normal spending habits).")

    importances = pd.Series(fused_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print("\nFeature importances (fused model):")
    print(importances.to_string())
