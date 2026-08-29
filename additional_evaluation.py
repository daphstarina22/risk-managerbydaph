"""
Additional evaluation rigor for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Three additions on top of the core classifier:
  1. Precision-recall curve plot (visual, for README/pitch)
  2. K-fold cross-validation (shows metrics aren't a lucky single split)
  3. Analyst workload projection (alerts/day a human would need to review)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

from fraud_classifier import generate_synthetic_data, engineer_features, FEATURES


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
# 2. K-fold cross-validation — proves the 0.967 wasn't a lucky split
# ---------------------------------------------------------------------
def cross_validate(df, n_splits=5):
    X = df[FEATURES]
    y = df["is_fraud"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        from sklearn.metrics import precision_score, recall_score
        fold_results.append({
            "fold": fold_idx,
            "pr_auc": average_precision_score(y_test, probs),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
        })

    results_df = pd.DataFrame(fold_results)
    print(f"\n{n_splits}-fold cross-validation results:")
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
    transaction volume, to answer: "how many alerts would a human review per day?"
    This is the practical, staffing-relevant version of the false-positive-cost story.
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
# Run everything
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("Generating data...")
    df = generate_synthetic_data()
    df = engineer_features(df)

    # Single split for the PR curve (matches fraud_classifier.py's approach)
    X = df[FEATURES]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- 1. Precision-Recall curve ---")
    plot_pr_curve(y_test, probs)

    print("\n--- 2. Cross-validation ---")
    cross_validate(df, n_splits=5)

    print("\n--- 3. Analyst workload projection ---")
    n_flagged = int(((probs >= 0.05)).sum())  # matches decision_layer's review threshold
    project_analyst_workload(
        n_flagged_in_test=n_flagged,
        test_size=len(X_test),
        daily_txn_volume=2_000_000,  # illustrative assumption, stated explicitly
    )
