"""
features.py
============
Feature engineering utilities for time series forecasting.

This module provides reusable functions to generate lag features,
rolling window statistics, calendar/date-based features, expanding
window statistics, and exponentially-weighted moving statistics from
a univariate (or multivariate) time-indexed pandas Series/DataFrame.

All functions:
    - Accept a pandas Series or DataFrame with a DatetimeIndex (or a
      column convertible to one).
    - Return a new DataFrame (do not mutate the input in place).
    - Raise informative errors on invalid input.
    - Use type hints and PEP8-compliant formatting.

Author: [Your Name]
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SeriesOrFrame = Union[pd.Series, pd.DataFrame]


def _validate_series(series: pd.Series, name: str = "series") -> None:
    """
    Validate that the input is a non-empty pandas Series with a
    DatetimeIndex (or an index that can reasonably represent time order).

    Parameters
    ----------
    series : pd.Series
        The series to validate.
    name : str
        Name used in error messages for clarity.

    Raises
    ------
    TypeError
        If `series` is not a pandas Series.
    ValueError
        If `series` is empty.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"'{name}' must be a pandas Series, got {type(series)}")
    if series.empty:
        raise ValueError(f"'{name}' must not be empty")


def create_lag_features(
    series: pd.Series,
    lags: Sequence[int],
    column_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create lagged versions of a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series (e.g., daily_ship_transits), indexed by date.
    lags : Sequence[int]
        List/tuple of positive integer lag periods to generate
        (e.g., [1, 2, 7] creates lag_1, lag_2, lag_7).
    column_name : Optional[str]
        Base name for the original series column in the output.
        Defaults to `series.name` or 'value' if unnamed.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original series plus one column per
        lag, named '{column_name}_lag_{k}'.

    Raises
    ------
    TypeError
        If `series` is not a pandas Series or `lags` is not iterable.
    ValueError
        If `series` is empty or any lag value is not a positive integer.

    Examples
    --------
    >>> df = create_lag_features(ts['daily_ship_transits'], lags=[1, 7])
    >>> df.columns.tolist()
    ['daily_ship_transits', 'daily_ship_transits_lag_1', 'daily_ship_transits_lag_7']
    """
    _validate_series(series, "series")

    if not isinstance(lags, (list, tuple)):
        raise TypeError(f"'lags' must be a list or tuple of ints, got {type(lags)}")

    for lag in lags:
        if not isinstance(lag, int) or lag <= 0:
            raise ValueError(f"All lag values must be positive integers, got {lag}")

    col_name = column_name or series.name or "value"
    result = pd.DataFrame({col_name: series})

    for lag in lags:
        result[f"{col_name}_lag_{lag}"] = series.shift(lag)

    logger.info("Created %d lag features for '%s': %s", len(lags), col_name, lags)
    return result


def create_rolling_features(
    series: pd.Series,
    windows: Sequence[int],
    stats: Sequence[str] = ("mean", "std"),
    column_name: Optional[str] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create rolling window statistics for a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series, indexed by date.
    windows : Sequence[int]
        Window sizes (in periods) over which to compute rolling stats
        (e.g., [3, 7, 14]).
    stats : Sequence[str]
        Statistics to compute for each window. Supported values:
        'mean', 'std', 'min', 'max', 'median'.
    column_name : Optional[str]
        Base name for the original series column. Defaults to
        `series.name` or 'value'.
    min_periods : Optional[int]
        Minimum number of observations required to compute a value.
        If None, defaults to the window size (i.e., no partial windows).

    Returns
    -------
    pd.DataFrame
        DataFrame with the original series plus one column per
        (window, stat) combination, named '{column_name}_roll{window}_{stat}'.

    Raises
    ------
    TypeError
        If inputs are not the expected types.
    ValueError
        If `series` is empty, a window is non-positive, or an
        unsupported statistic is requested.

    Examples
    --------
    >>> df = create_rolling_features(ts['daily_ship_transits'], windows=[7],
    ...                               stats=['mean', 'std'])
    >>> df.columns.tolist()
    ['daily_ship_transits', 'daily_ship_transits_roll7_mean', 'daily_ship_transits_roll7_std']
    """
    _validate_series(series, "series")

    supported_stats = {"mean", "std", "min", "max", "median"}
    for stat in stats:
        if stat not in supported_stats:
            raise ValueError(
                f"Unsupported stat '{stat}'. Supported stats: {supported_stats}"
            )

    for window in windows:
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"All window sizes must be positive integers, got {window}")

    col_name = column_name or series.name or "value"
    result = pd.DataFrame({col_name: series})

    for window in windows:
        mp = min_periods if min_periods is not None else window
        rolling_obj = series.rolling(window=window, min_periods=mp)
        for stat in stats:
            result[f"{col_name}_roll{window}_{stat}"] = getattr(rolling_obj, stat)()

    logger.info(
        "Created rolling features for '%s' | windows=%s | stats=%s",
        col_name, windows, stats,
    )
    return result


def create_date_features(
    index: Union[pd.DatetimeIndex, pd.Series],
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Create calendar-based features from a datetime index or series.

    Parameters
    ----------
    index : pd.DatetimeIndex or pd.Series
        Datetime index (or a series of datetimes) from which calendar
        features are derived.
    features : Optional[Sequence[str]]
        Which calendar features to generate. Supported values:
        'day_of_week', 'day_of_month', 'week_of_year', 'month',
        'quarter', 'is_weekend', 'is_month_start', 'is_month_end'.
        If None, all supported features are generated.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed the same as `index`, with one column per
        requested calendar feature.

    Raises
    ------
    TypeError
        If `index` cannot be converted to a DatetimeIndex.
    ValueError
        If `index` is empty or an unsupported feature name is requested.

    Examples
    --------
    >>> df = create_date_features(ts.index, features=['day_of_week', 'is_weekend'])
    >>> df.columns.tolist()
    ['day_of_week', 'is_weekend']
    """
    if isinstance(index, pd.Series):
        try:
            index = pd.DatetimeIndex(pd.to_datetime(index))
        except Exception as exc:
            raise TypeError(
                f"Could not convert 'index' Series to DatetimeIndex: {exc}"
            ) from exc
    elif isinstance(index, pd.DatetimeIndex):
        pass
    else:
        try:
            index = pd.DatetimeIndex(pd.to_datetime(index))
        except Exception as exc:
            raise TypeError(
                f"'index' must be a DatetimeIndex or convertible to one: {exc}"
            ) from exc

    if len(index) == 0:
        raise ValueError("'index' must not be empty")

    all_features = {
        "day_of_week": lambda idx: idx.dayofweek,
        "day_of_month": lambda idx: idx.day,
        "week_of_year": lambda idx: idx.isocalendar().week.values,
        "month": lambda idx: idx.month,
        "quarter": lambda idx: idx.quarter,
        "is_weekend": lambda idx: (idx.dayofweek >= 5).astype(int),
        "is_month_start": lambda idx: idx.is_month_start.astype(int),
        "is_month_end": lambda idx: idx.is_month_end.astype(int),
    }

    selected = features if features is not None else list(all_features.keys())

    for feat in selected:
        if feat not in all_features:
            raise ValueError(
                f"Unsupported date feature '{feat}'. "
                f"Supported features: {list(all_features.keys())}"
            )

    result = pd.DataFrame(index=index)
    for feat in selected:
        result[feat] = all_features[feat](index)

    logger.info("Created %d date features: %s", len(selected), selected)
    return result


def create_expanding_features(
    series: pd.Series,
    stats: Sequence[str] = ("mean", "std"),
    column_name: Optional[str] = None,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Create expanding (cumulative) window statistics for a time series.

    Expanding statistics use all observations up to and including the
    current point, useful for capturing "all history so far" signals
    without lookahead leakage.

    Parameters
    ----------
    series : pd.Series
        Input time series, indexed by date.
    stats : Sequence[str]
        Statistics to compute. Supported values:
        'mean', 'std', 'min', 'max', 'sum'.
    column_name : Optional[str]
        Base name for the original series column. Defaults to
        `series.name` or 'value'.
    min_periods : int
        Minimum number of observations required to compute a value.
        Defaults to 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original series plus one column per
        statistic, named '{column_name}_expanding_{stat}'.

    Raises
    ------
    TypeError
        If `series` is not a pandas Series.
    ValueError
        If `series` is empty, `min_periods` is invalid, or an
        unsupported statistic is requested.

    Examples
    --------
    >>> df = create_expanding_features(ts['daily_ship_transits'], stats=['mean'])
    >>> df.columns.tolist()
    ['daily_ship_transits', 'daily_ship_transits_expanding_mean']
    """
    _validate_series(series, "series")

    supported_stats = {"mean", "std", "min", "max", "sum"}
    for stat in stats:
        if stat not in supported_stats:
            raise ValueError(
                f"Unsupported stat '{stat}'. Supported stats: {supported_stats}"
            )

    if not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError(f"'min_periods' must be a positive integer, got {min_periods}")

    col_name = column_name or series.name or "value"
    result = pd.DataFrame({col_name: series})

    expanding_obj = series.expanding(min_periods=min_periods)
    for stat in stats:
        result[f"{col_name}_expanding_{stat}"] = getattr(expanding_obj, stat)()

    logger.info("Created expanding features for '%s': %s", col_name, stats)
    return result


def create_ewm_features(
    series: pd.Series,
    spans: Sequence[int],
    stats: Sequence[str] = ("mean",),
    column_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create exponentially-weighted moving (EWM) statistics for a time series.

    EWM features give more weight to recent observations, controlled by
    the `span` parameter (larger span = slower decay / more smoothing).

    Parameters
    ----------
    series : pd.Series
        Input time series, indexed by date.
    spans : Sequence[int]
        Span values for exponential weighting (e.g., [3, 7, 14]).
    stats : Sequence[str]
        Statistics to compute. Supported values: 'mean', 'std'.
        Note: EWM std requires `bias=False` (default) and is only
        well-defined with sufficient data.
    column_name : Optional[str]
        Base name for the original series column. Defaults to
        `series.name` or 'value'.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original series plus one column per
        (span, stat) combination, named '{column_name}_ewm{span}_{stat}'.

    Raises
    ------
    TypeError
        If `series` is not a pandas Series.
    ValueError
        If `series` is empty, a span is non-positive, or an
        unsupported statistic is requested.

    Examples
    --------
    >>> df = create_ewm_features(ts['daily_ship_transits'], spans=[7], stats=['mean'])
    >>> df.columns.tolist()
    ['daily_ship_transits', 'daily_ship_transits_ewm7_mean']
    """
    _validate_series(series, "series")

    supported_stats = {"mean", "std"}
    for stat in stats:
        if stat not in supported_stats:
            raise ValueError(
                f"Unsupported stat '{stat}'. Supported stats: {supported_stats}"
            )

    for span in spans:
        if not isinstance(span, int) or span <= 0:
            raise ValueError(f"All span values must be positive integers, got {span}")

    col_name = column_name or series.name or "value"
    result = pd.DataFrame({col_name: series})

    for span in spans:
        ewm_obj = series.ewm(span=span, adjust=False)
        for stat in stats:
            result[f"{col_name}_ewm{span}_{stat}"] = getattr(ewm_obj, stat)()

    logger.info("Created EWM features for '%s': spans=%s, stats=%s", col_name, spans, stats)
    return result


def build_feature_matrix(
    series: pd.Series,
    lags: Sequence[int] = (1, 2, 7),
    rolling_windows: Sequence[int] = (7, 14),
    ewm_spans: Sequence[int] = (7,),
    date_features: Optional[Sequence[str]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convenience function: build a full feature matrix combining lag,
    rolling, EWM, and date features for a single target series.

    Parameters
    ----------
    series : pd.Series
        Input time series with a DatetimeIndex.
    lags : Sequence[int]
        Lag periods to include.
    rolling_windows : Sequence[int]
        Rolling window sizes to include (mean and std computed for each).
    ewm_spans : Sequence[int]
        EWM spans to include (mean computed for each).
    date_features : Optional[Sequence[str]]
        Calendar features to include. Defaults to
        ['day_of_week', 'is_weekend'] if None.
    dropna : bool
        If True (default), drop rows containing any NaN values
        introduced by lagging/rolling — typical for supervised
        learning setups (e.g., XGBoost).

    Returns
    -------
    pd.DataFrame
        Combined feature matrix with the original series, lag,
        rolling, EWM, and date features.

    Examples
    --------
    >>> X = build_feature_matrix(ts['daily_ship_transits'])
    >>> X.shape[1] > 1
    True
    """
    _validate_series(series, "series")

    if date_features is None:
        date_features = ["day_of_week", "is_weekend"]

    lag_df = create_lag_features(series, lags=list(lags))
    roll_df = create_rolling_features(
        series, windows=list(rolling_windows), stats=["mean", "std"]
    )
    ewm_df = create_ewm_features(series, spans=list(ewm_spans), stats=["mean"])
    date_df = create_date_features(series.index, features=list(date_features))

    combined = lag_df.join(
        roll_df.drop(columns=[series.name or "value"]), how="left"
    ).join(
        ewm_df.drop(columns=[series.name or "value"]), how="left"
    ).join(date_df, how="left")

    if dropna:
        before = len(combined)
        combined = combined.dropna()
        logger.info("Dropped %d rows with NaNs (from lag/rolling warmup)", before - len(combined))

    return combined


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a small synthetic example time series for demonstration
    rng = pd.date_range("2026-01-01", periods=30, freq="D")
    example_series = pd.Series(
        np.concatenate([np.full(15, 100.0), np.full(15, 5.0)]),
        index=rng,
        name="daily_ship_transits",
    )

    print("=== Lag Features ===")
    print(create_lag_features(example_series, lags=[1, 7]).head(10))

    print("\n=== Rolling Features ===")
    print(create_rolling_features(example_series, windows=[3, 7]).head(10))

    print("\n=== Date Features ===")
    print(create_date_features(example_series.index, features=["day_of_week", "is_weekend"]).head(5))

    print("\n=== Expanding Features ===")
    print(create_expanding_features(example_series).head(10))

    print("\n=== EWM Features ===")
    print(create_ewm_features(example_series, spans=[3, 7]).head(10))

    print("\n=== Full Feature Matrix (ML-ready) ===")
    feature_matrix = build_feature_matrix(example_series)
    print(feature_matrix.head())
    print("Shape:", feature_matrix.shape)
