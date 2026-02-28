"""Data preprocessing: scaling, target creation, windowing, train/test split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Container for train/validation/test splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: RobustScaler


@dataclass
class SequenceData:
    """Container for LSTM sequence data."""

    X_train: np.ndarray  # (n_samples, seq_len, n_features)
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: RobustScaler


class Preprocessor:
    """Transforms feature DataFrames into ML-ready arrays."""

    def __init__(self, threshold: float = 0.005):
        """
        Args:
            threshold: Price change threshold for target labels.
                       >threshold = BUY (2), <-threshold = SELL (0), else HOLD (1).
        """
        self.threshold = threshold

    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """Create classification target: 0=SELL, 1=HOLD, 2=BUY.

        Based on forward return over `horizon` periods.
        """
        df = df.copy()
        forward_return = df["close"].shift(-horizon) / df["close"] - 1

        df["target"] = 1  # HOLD
        df.loc[forward_return > self.threshold, "target"] = 2  # BUY
        df.loc[forward_return < -self.threshold, "target"] = 0  # SELL

        # Drop rows where target can't be computed
        df = df.iloc[:-horizon]

        return df

    def prepare_tabular(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> SplitData:
        """Prepare tabular data for XGBoost-style models.

        Walk-forward split (temporal, not random).
        """
        df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X = df[feature_cols].values
        y = df["target"].values.astype(int)

        # Fit scaler on train data only
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X[:train_end])
        X_val = scaler.transform(X[train_end:val_end])
        X_test = scaler.transform(X[val_end:])

        logger.info(
            "tabular_data_prepared",
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
            n_features=len(feature_cols),
        )

        return SplitData(
            X_train=X_train,
            y_train=y[:train_end],
            X_val=X_val,
            y_val=y[train_end:val_end],
            X_test=X_test,
            y_test=y[val_end:],
            feature_names=feature_cols,
            scaler=scaler,
        )

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_length: int = 60,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> SequenceData:
        """Prepare sequential data for LSTM-style models.

        Creates rolling windows of `seq_length` steps.
        """
        df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

        X_raw = df[feature_cols].values
        y_raw = df["target"].values.astype(int)

        # Scale first
        n = len(df)
        train_end = int(n * train_ratio)

        scaler = RobustScaler()
        scaler.fit(X_raw[:train_end])
        X_scaled = scaler.transform(X_raw)

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i - seq_length : i])
            y_seq.append(y_raw[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Split (adjust for sequence offset)
        adj_train_end = max(0, train_end - seq_length)
        adj_val_end = max(0, int(n * (train_ratio + val_ratio)) - seq_length)

        logger.info(
            "sequence_data_prepared",
            train_size=adj_train_end,
            val_size=adj_val_end - adj_train_end,
            test_size=len(X_seq) - adj_val_end,
            seq_length=seq_length,
            n_features=len(feature_cols),
        )

        return SequenceData(
            X_train=X_seq[:adj_train_end],
            y_train=y_seq[:adj_train_end],
            X_val=X_seq[adj_train_end:adj_val_end],
            y_val=y_seq[adj_train_end:adj_val_end],
            X_test=X_seq[adj_val_end:],
            y_test=y_seq[adj_val_end:],
            feature_names=feature_cols,
            scaler=scaler,
        )

    @staticmethod
    def save_scaler(scaler: RobustScaler, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, path)

    @staticmethod
    def load_scaler(path: Path) -> RobustScaler:
        import joblib

        return joblib.load(path)


preprocessor = Preprocessor()
