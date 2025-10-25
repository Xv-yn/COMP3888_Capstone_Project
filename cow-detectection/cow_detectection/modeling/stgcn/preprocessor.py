"""
Preprocessor for ST-GCN datasets.

Handles data preparation for both 1-stream and 2-stream ST-GCN models.
- Loads HRNet keypoints or pre-saved .pkl features.
- Converts them into ST-GCN input format (N, C, T, V).
- Supports optional two-stream mode (pose + motion).
- Provides scaling with sklearn StandardScaler.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils import data

from cow_detectection.modeling.base import BasePreprocessor


class KeypointPreprocessor(BasePreprocessor):
    """
    Preprocessor for HRNet keypoints → ST-GCN input format.

    Attributes:
        X_train_, X_test_, y_train_, y_test_: processed splits
        scaler_: StandardScaler used for normalization
    """

    def __init__(self, two_stream: bool = False):
        super().__init__()
        self.two_stream = two_stream

    def get_data_and_labels(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, label_column: str = "label"
    ):
        """
        Converts train/test DataFrames into ST-GCN input format.

        Args:
            df_train: training dataset with keypoints + labels
            df_test: testing dataset with keypoints + labels
            label_column: column containing labels

        Returns:
            None (sets class attributes)
        """
        self.y_train_ = df_train[label_column].values
        self.y_test_ = df_test[label_column].values

        self.X_train_ = self._build_features(df_train)
        self.X_test_ = self._build_features(df_test)

    @staticmethod
    def load_dataset(data_files, batch_size: int, split_size: float = 0.0):
        """
        Load data files into torch DataLoader with/without splitting train-test.

        Args:
            data_files: list of paths to .pkl files, each containing (features, labels).
                        Features expected in shape (N, T, V, C).
            batch_size: batch size for DataLoader.
            split_size: float in [0,1], fraction of data for validation split.

        Returns:
            train_loader, valid_loader
        """
        features, labels = [], []
        for fil in data_files:
            with open(fil, "rb") as f:
                fts, lbs = pickle.load(f)
                features.append(fts)
                labels.append(lbs)

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Convert (N, T, V, C) -> (N, C, T, V)
        features = torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2)
        labels = torch.tensor(labels, dtype=torch.long)

        if split_size > 0:
            x_train, x_valid, y_train, y_valid = train_test_split(
                features, labels, test_size=split_size, random_state=9, stratify=labels
            )
            train_set = data.TensorDataset(x_train, y_train)
            valid_set = data.TensorDataset(x_valid, y_valid)
            train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = data.DataLoader(valid_set, batch_size=batch_size)
            return train_loader, valid_loader

        train_set = data.TensorDataset(features, labels)
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return train_loader, None

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Builds feature arrays for 1-stream or 2-stream input.

        Returns:
            features: np.ndarray shaped (N, C, T, V)
        """
        pose = self._reshape_pose(df)

        if not self.two_stream:
            return pose

        motion = self._compute_motion(pose)
        return np.concatenate([pose, motion], axis=1)

    def _compute_motion(self, pose: np.ndarray) -> np.ndarray:
        """
        Computes motion features (Δx, Δy) from pose tensor.

        Args:
            pose: (N, 3, T, V)

        Returns:
            motion: (N, 2, T, V)
        """
        x, y = pose[:, 0, :, :], pose[:, 1, :, :]
        dx = np.diff(x, axis=1, prepend=0)
        dy = np.diff(y, axis=1, prepend=0)

        motion = np.stack([dx, dy], axis=1)
        return motion

    def scale(self, scaler: StandardScaler, X_train: np.ndarray, X_test: np.ndarray):
        """
        Scales train/test features using sklearn scaler, avoiding data leakage.

        Args:
            scaler: sklearn scaler (e.g., StandardScaler).
            X_train: (N, C, T, V).
            X_test: (N, C, T, V).

        Returns:
            Tuple of scaled train/test arrays with same shape.
        """
        N_train, C, T, V = X_train.shape
        N_test = X_test.shape[0]

        X_train_flat = X_train.reshape(N_train, -1)
        X_test_flat = X_test.reshape(N_test, -1)

        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        self.scaler_ = scaler

        return (X_train_scaled.reshape(N_train, C, T, V), X_test_scaled.reshape(N_test, C, T, V))
