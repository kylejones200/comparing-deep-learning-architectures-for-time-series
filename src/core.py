"""Core functions for comparing deep learning architectures for time series."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def create_lagged_features(
    data: np.ndarray, lag: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged features for time series."""
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def prepare_data_for_lstm(
    data: pd.Series, lag: int = 10, train_size: float = 0.8
) -> Tuple:
    """Prepare data for LSTM model."""
    scaler = MinMaxScaler()
    values = data.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    X, y = create_lagged_features(scaled.flatten(), lag)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    train_size_int = int(len(X) * train_size)
    X_train, X_test = X[:train_size_int], X[train_size_int:]
    y_train, y_test = y[:train_size_int], y[train_size_int:]
    return X_train, X_test, y_train, y_test, scaler


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate model performance metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def plot_architecture_comparison(
    results: Dict[str, np.ndarray], actual: np.ndarray, title: str, output_path: Path
, plot: bool = False):
    """Plot comparison of different architectures"""
    if not plot:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual, label="Actual", color="#4A90A4", linewidth=1.2)
    colors = ["#D4A574", "#8B6F9E", "#A8C5A0", "#E8A87C"]
    for i, (name, pred) in enumerate(results.items()):
        ax.plot(
            pred,
            label=name,
            color=colors[i % len(colors)],
            linewidth=1.2,
            alpha=0.7,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
