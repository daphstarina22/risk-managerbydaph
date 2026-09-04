"""
Behavioral baseline profiler for the fraud-spike detector.
Razorpay Buildathon — AI Risk Manager track.

Computes user-specific spending baselines strictly on training data to prevent
data leakage across train/validation/test splits and between folds. Provides
population-level fallback priors for unseen users during inference.
"""

import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class UserProfiler:
    """
    Learns per-user historical spending baselines (mean, std) from training data
    and computes behavioral deviation features (amount_zscore).
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.user_stats: Dict[int, Dict[str, float]] = {}
        self.global_mean: float = 0.0
        self.global_std: float = 1.0
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "UserProfiler":
        """
        Fits per-user baselines and global population priors on training transactions.
        Must only be called on training data.
        """
        if "amount" not in df.columns:
            raise ValueError("Dataframe must contain 'amount' column.")

        # Compute global priors for unseen users
        self.global_mean = float(df["amount"].mean())
        self.global_std = float(df["amount"].std())
        if np.isnan(self.global_std) or self.global_std < self.eps:
            self.global_std = max(self.eps, 1.0)

        # Compute per-user mean and std
        if "user_id" in df.columns:
            grouped = df.groupby("user_id")["amount"].agg(["mean", "std"])
            stats_dict = {}
            for uid, row in grouped.iterrows():
                mean_val = float(row["mean"])
                std_val = float(row["std"]) if not np.isnan(row["std"]) and row["std"] > self.eps else self.eps
                stats_dict[int(uid)] = {"mean": mean_val, "std": std_val}
            self.user_stats = stats_dict
        else:
            self.user_stats = {}

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms transaction data by calculating amount_zscore using learned baselines.
        Unseen users seamlessly receive population-level fallback priors.
        """
        if not self.is_fitted:
            raise RuntimeError("UserProfiler must be fitted before calling transform().")

        df = df.copy()

        if "user_id" in df.columns and self.user_stats:
            stats_df = pd.DataFrame.from_dict(self.user_stats, orient="index")
            stats_df.index.name = "user_id"

            df = df.join(stats_df, on="user_id", rsuffix="_user")
            df["mean"] = df["mean"].fillna(self.global_mean)
            df["std"] = df["std"].fillna(self.global_std).replace(0, self.eps)

            df["amount_zscore"] = (df["amount"] - df["mean"]) / df["std"]
            df.drop(columns=["mean", "std"], inplace=True)
        else:
            # Fallback to global priors
            df["amount_zscore"] = (df["amount"] - self.global_mean) / self.global_std

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to fit and transform on training data."""
        return self.fit(df).transform(df)

    def transform_single(self, txn: Dict[str, Any]) -> float:
        """
        Calculates amount_zscore for a single transaction dictionary.
        If amount_zscore is already present in txn, returns it directly.
        Otherwise, computes deviation from the user's learned baseline or global prior.
        """
        if "amount_zscore" in txn and txn["amount_zscore"] is not None:
            return float(txn["amount_zscore"])

        if not self.is_fitted:
            raise RuntimeError("UserProfiler must be fitted before calling transform_single().")

        amount = float(txn["amount"])
        user_id = int(txn.get("user_id", -1))

        if user_id in self.user_stats:
            user_mean = self.user_stats[user_id]["mean"]
            user_std = self.user_stats[user_id]["std"]
        else:
            user_mean = self.global_mean
            user_std = self.global_std

        std = user_std if user_std > self.eps else self.eps
        return round((amount - user_mean) / std, 4)

    def save(self, path: str) -> None:
        """Saves learned profile baselines to a JSON file."""
        data = {
            "eps": self.eps,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "user_stats": self.user_stats,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UserProfiler":
        """Loads profile baselines from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        profiler = cls(eps=data.get("eps", 1e-3))
        profiler.global_mean = float(data["global_mean"])
        profiler.global_std = float(data["global_std"])
        profiler.user_stats = {int(k): v for k, v in data["user_stats"].items()}
        profiler.is_fitted = True
        return profiler
