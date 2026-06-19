"""
evaluate.py
===========
Evaluation utilities for time series forecasting models.

Provides:
- Core metric functions (MAE, RMSE, MAPE, directional accuracy)
- Model comparison table builder
- Walk-forward (time series) cross-validation
- Residual analysis helpers

Author: [Your Name]
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Zero values in `y_true` are excluded to avoid division-by-zero.
    Returns NaN if all y_true values are zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        logger.warning("All y_true values are zero; MAPE undefined → NaN")
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (%) — measures whether the forecast correctly
    predicts the direction of change (up/down) relative to the previous step.

    Parameters
    ----------
    y_true, y_pred : array-like
        Must have at least 2 elements.

    Returns
    -------
    float
        Percentage of periods where the predicted direction matches actual.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        logger.warning("Directional accuracy requires >= 2 observations")
        return np.nan
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == pred_dir) * 100)


def compute_all_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and Directional Accuracy for a forecast.

    Parameters
    ----------
    y_true : pd.Series
        Actual observed values (test set).
    y_pred : pd.Series
        Forecasted values for the same period.
    model_name : str
        Label used in log output.

    Returns
    -------
    Dict[str, float]
        {'Model', 'MAE', 'RMSE', 'MAPE', 'DirectionalAccuracy'}
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    metrics = {
        "Model": model_name,
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": mape(yt, yp),
        "DirectionalAcc(%)": directional_accuracy(yt, yp),
    }
    logger.info(
        "[%s] MAE=%.4f | RMSE=%.4f | MAPE=%.2f%% | DirAcc=%.1f%%",
        model_name, metrics["MAE"], metrics["RMSE"],
        metrics["MAPE"], metrics["DirectionalAcc(%)"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Build a ranked model comparison table from a list of metric dicts.

    Parameters
    ----------
    results : List[Dict[str, float]]
        Each dict should have keys: 'Model', 'MAE', 'RMSE', 'MAPE',
        'DirectionalAcc(%)' — as returned by `compute_all_metrics`.

    Returns
    -------
    pd.DataFrame
        Sorted by RMSE (ascending), with a Rank column added.
        Best model per metric is highlighted in log output.
    """
    df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    for col in ["MAE", "RMSE", "MAPE"]:
        df[col] = df[col].round(4)
    df["DirectionalAcc(%)"] = df["DirectionalAcc(%)"].round(1)

    best_rmse = df.loc[df["RMSE"].idxmin(), "Model"]
    best_mape = df.loc[df["MAPE"].idxmin(), "Model"]
    logger.info("Best RMSE: %s | Best MAPE: %s", best_rmse, best_mape)

    return df


def plot_comparison_table(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot a horizontal bar chart comparing model RMSE values.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `build_comparison_table`.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(df) * 0.6)))

    for ax, metric, color in zip(axes, ["RMSE", "MAE", "MAPE"],
                                  ["steelblue", "coral", "seagreen"]):
        bars = ax.barh(df["Model"][::-1], df[metric][::-1], color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(f"Model Comparison — {metric}", fontsize=12)
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Forecasting Model Benchmark Results", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Saved comparison chart to '%s'", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(
    series: pd.Series,
    model_fn: Callable,
    n_splits: int = 5,
    test_size: int = 10,
    min_train_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Walk-forward (rolling-origin) time series cross-validation.

    At each fold, the model is trained on all data up to the split
    point and evaluated on the next `test_size` periods. This is the
    correct way to cross-validate time series models without leakage.

    Parameters
    ----------
    series : pd.Series
        Full time series.
    model_fn : Callable
        A function that takes (train: pd.Series, steps: int) and returns
        a pd.Series of predictions. Example::

            def model_fn(train, steps):
                m = ARIMAForecaster(order=(1,1,1)).fit(train)
                return m.predict(steps)

    n_splits : int
        Number of cross-validation folds.
    test_size : int
        Number of periods per test fold.
    min_train_size : Optional[int]
        Minimum training observations required. If None, uses the
        minimum that leaves enough data for `n_splits` folds.

    Returns
    -------
    pd.DataFrame
        One row per fold with columns:
        'Fold', 'TrainSize', 'TestStart', 'TestEnd', 'MAE', 'RMSE', 'MAPE'.
    """
    n = len(series)
    required = test_size * n_splits
    if min_train_size is None:
        min_train_size = n - required

    if min_train_size < 10:
        raise ValueError(
            f"Not enough data for {n_splits} folds of size {test_size}. "
            f"Reduce n_splits or test_size."
        )

    fold_records = []

    for fold in range(1, n_splits + 1):
        test_end = n - test_size * (n_splits - fold)
        test_start = test_end - test_size
        train = series.iloc[:test_start]
        test = series.iloc[test_start:test_end]

        try:
            pred = model_fn(train, test_size)
            pred_values = np.asarray(pred, dtype=float)
            record = {
                "Fold": fold,
                "TrainSize": len(train),
                "TestStart": test.index[0].date() if hasattr(test.index[0], "date") else test.index[0],
                "TestEnd": test.index[-1].date() if hasattr(test.index[-1], "date") else test.index[-1],
                "MAE": mae(test.values, pred_values),
                "RMSE": rmse(test.values, pred_values),
                "MAPE": mape(test.values, pred_values),
            }
            logger.info(
                "Fold %d/%d | Train=%d | MAE=%.3f | RMSE=%.3f | MAPE=%.2f%%",
                fold, n_splits, len(train),
                record["MAE"], record["RMSE"], record["MAPE"],
            )
        except Exception as exc:
            logger.error("Fold %d failed: %s", fold, exc)
            record = {
                "Fold": fold, "TrainSize": len(train),
                "TestStart": None, "TestEnd": None,
                "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
            }

        fold_records.append(record)

    cv_df = pd.DataFrame(fold_records)
    logger.info(
        "CV Summary — Mean RMSE: %.3f (±%.3f)",
        cv_df["RMSE"].mean(), cv_df["RMSE"].std(),
    )
    return cv_df


def plot_cv_results(cv_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot walk-forward CV RMSE across folds.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Output of `walk_forward_cv`.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cv_df["Fold"], cv_df["RMSE"], marker="o", color="steelblue", linewidth=2)
    ax.axhline(cv_df["RMSE"].mean(), color="red", linestyle="--",
               label=f"Mean RMSE = {cv_df['RMSE'].mean():.3f}")
    ax.fill_between(
        cv_df["Fold"],
        cv_df["RMSE"].mean() - cv_df["RMSE"].std(),
        cv_df["RMSE"].mean() + cv_df["RMSE"].std(),
        alpha=0.15, color="red", label="±1 Std Dev",
    )
    ax.set_title("Walk-Forward Cross-Validation — RMSE per Fold", fontsize=13)
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a 4-panel residual analysis dashboard:
    1. Residuals over time
    2. Residual histogram + KDE
    3. Q-Q plot
    4. Actual vs Predicted scatter

    Parameters
    ----------
    y_true : pd.Series
        Actual values (test set).
    y_pred : pd.Series
        Forecasted values.
    model_name : str
        Used in plot titles.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    from scipy import stats

    residuals = pd.Series(
        np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
        index=y_true.index,
        name="residuals",
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=14)

    # 1. Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, marker="o",
                    color="steelblue", linewidth=1.5)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram + KDE
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color="coral")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual")

    # 3. Q-Q plot
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normality Check)")

    # 4. Actual vs Predicted
    axes[1, 1].scatter(y_true.values, y_pred.values if hasattr(y_pred, 'values') else y_pred,
                       alpha=0.7, color="seagreen")
    min_val = min(y_true.min(), np.min(y_pred))
    max_val = max(y_true.max(), np.max(y_pred))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
    axes[1, 1].set_title("Actual vs Predicted")
    axes[1, 1].set_xlabel("Actual")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Saved residual plot to '%s'", save_path)
    plt.show()

    # Summary stats
    print(f"\nResidual Statistics — {model_name}:")
    print(f"  Mean  : {residuals.mean():.4f}  (ideally ≈ 0)")
    print(f"  Std   : {residuals.std():.4f}")
    print(f"  Skew  : {residuals.skew():.4f}  (ideally ≈ 0)")
    print(f"  Kurt  : {residuals.kurtosis():.4f}  (ideally ≈ 0 for normal)")


def plot_forecast_vs_actual(
    train: pd.Series,
    test: pd.Series,
    forecasts: Dict[str, pd.Series],
    conf_intervals: Optional[Dict[str, pd.DataFrame]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training data, test actuals, and multiple model forecasts
    on a single chart with optional confidence intervals.

    Parameters
    ----------
    train : pd.Series
        Training time series.
    test : pd.Series
        Actual test values.
    forecasts : Dict[str, pd.Series]
        Model name → forecast Series mapping.
    conf_intervals : Optional[Dict[str, pd.DataFrame]]
        Model name → DataFrame with 'lower' and 'upper' columns.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(16, 7))

    # Training history (last 30 days for readability)
    plt.plot(train.index[-30:], train.values[-30:],
             color="gray", label="Train (last 30d)", linewidth=1.5, alpha=0.6)

    # Actual test
    plt.plot(test.index, test.values,
             color="black", linewidth=2.5, label="Actual (Test)", zorder=5)

    # Forecasts
    colors = ["steelblue", "coral", "seagreen", "darkorange",
              "purple", "crimson", "teal", "olive"]
    for (name, pred), color in zip(forecasts.items(), colors):
        plt.plot(test.index, np.asarray(pred)[:len(test)],
                 "--", linewidth=1.8, label=name, color=color)
        if conf_intervals and name in conf_intervals:
            ci = conf_intervals[name]
            plt.fill_between(
                test.index,
                ci.iloc[:, 0].values[:len(test)],
                ci.iloc[:, 1].values[:len(test)],
                alpha=0.1, color=color,
            )

    plt.title("Forecast vs Actual — Model Comparison", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Daily Ship Transits")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Saved forecast chart to '%s'", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    rng = pd.date_range("2026-01-01", periods=50, freq="D")
    y_true = pd.Series(np.concatenate([np.full(35, 100.0), np.full(15, 6.0)]), index=rng)

    # Simulate two model predictions
    y_pred_arima = pd.Series(np.concatenate([np.full(35, 100.0), np.full(15, 7.5)]), index=rng)
    y_pred_naive = pd.Series(np.concatenate([np.full(35, 100.0), np.full(15, 100.0)]), index=rng)

    test_true = y_true.iloc[-15:]
    test_arima = y_pred_arima.iloc[-15:]
    test_naive = y_pred_naive.iloc[-15:]

    results = [
        compute_all_metrics(test_true, test_arima, "ARIMA"),
        compute_all_metrics(test_true, test_naive, "Naive"),
    ]

    table = build_comparison_table(results)
    print("\nModel Comparison Table:")
    print(table.to_string(index=False))