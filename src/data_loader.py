"""
data_loader.py
==============
Data loading and preprocessing utilities for the Hormuz Strait
Shipping Disruption Forecasting project.

Author: [Your Name]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Default paths (relative to project root)
RAW_DATA_PATH = Path("data/raw/strait_of_hormuz_shipping_disruption_2026.csv")
PROCESSED_DATA_PATH = Path("data/processed/hormuz_processed.csv")

TARGET_COL = "daily_ship_transits"
DATE_COL = "date"


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw CSV dataset and perform minimal type coercion.

    Parameters
    ----------
    filepath : Optional[Path]
        Path to the raw CSV file. Defaults to RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with `date` parsed as datetime and sorted
        chronologically.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing from the CSV.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at '{path}'. "
            "Place the raw CSV in data/raw/ and retry."
        )

    logger.info("Loading raw data from '%s'", path)
    df = pd.read_csv(path)

    required_cols = {DATE_COL, TARGET_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from CSV: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    logger.info(
        "Loaded %d rows x %d columns | Date range: %s to %s",
        df.shape[0], df.shape[1],
        df[DATE_COL].min().date(),
        df[DATE_COL].max().date(),
    )
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run data quality checks and log warnings for any issues found.

    Checks performed:
    - Duplicate rows
    - Duplicate dates
    - Missing values in target column
    - Chronological ordering

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (output of `load_raw_data`).

    Returns
    -------
    pd.DataFrame
        The same DataFrame (unmodified); issues are logged as warnings.

    Raises
    ------
    ValueError
        If the target column contains missing values (breaks time series).
    """
    logger.info("Running data validation checks...")

    # Duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        logger.warning("%d fully duplicated rows detected", n_dup)
    else:
        logger.info("No duplicate rows found ✓")

    # Duplicate dates
    n_dup_dates = df[DATE_COL].duplicated().sum()
    if n_dup_dates > 0:
        logger.warning("%d duplicated dates detected", n_dup_dates)
    else:
        logger.info("No duplicate dates found ✓")

    # Missing values in target
    n_missing_target = df[TARGET_COL].isnull().sum()
    if n_missing_target > 0:
        raise ValueError(
            f"Target column '{TARGET_COL}' has {n_missing_target} missing values. "
            "Impute or remove before modeling."
        )
    else:
        logger.info("No missing values in target column ✓")

    # Overall missing values
    missing_summary = df.isnull().sum()
    cols_with_missing = missing_summary[missing_summary > 0]
    if not cols_with_missing.empty:
        logger.warning(
            "Columns with missing values:\n%s", cols_with_missing.to_string()
        )
    else:
        logger.info("No missing values in any column ✓")

    # Chronological order
    if not df[DATE_COL].is_monotonic_increasing:
        logger.warning("Data is not sorted chronologically — re-sorting")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    else:
        logger.info("Data is chronologically ordered ✓")

    return df


def get_time_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract the target time series as a date-indexed pandas Series.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with `date` and `daily_ship_transits` columns.

    Returns
    -------
    pd.Series
        Target series indexed by `date` (DatetimeIndex).
    """
    ts = df.set_index(DATE_COL)[TARGET_COL]
    ts.index.freq = pd.infer_freq(ts.index)
    logger.info("Extracted time series: %d observations", len(ts))
    return ts


def train_test_split_ts(
    series: pd.Series,
    test_size: int = 15,
) -> Tuple[pd.Series, pd.Series]:
    """
    Chronological train/test split for time series (no shuffling).

    Parameters
    ----------
    series : pd.Series
        Full time series.
    test_size : int
        Number of periods to hold out as the test set (from the end).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (train, test) series in chronological order.

    Raises
    ------
    ValueError
        If `test_size` >= len(series) or `test_size` <= 0.
    """
    if test_size <= 0 or test_size >= len(series):
        raise ValueError(
            f"'test_size' must be between 1 and {len(series) - 1}, got {test_size}"
        )

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    logger.info(
        "Train: %d obs (%s → %s) | Test: %d obs (%s → %s)",
        len(train), train.index[0].date(), train.index[-1].date(),
        len(test), test.index[0].date(), test.index[-1].date(),
    )
    return train, test


def get_exog_features(
    df: pd.DataFrame,
    feature_cols: list,
    index_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Extract exogenous feature columns as a date-indexed DataFrame,
    ready for use in SARIMAX.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame.
    feature_cols : list
        List of column names to use as exogenous regressors.
    index_col : str
        Column to use as the index. Defaults to `date`.

    Returns
    -------
    pd.DataFrame
        DataFrame of exogenous features indexed by date.

    Raises
    ------
    ValueError
        If any requested feature column is not present in `df`.
    """
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_cols}")

    exog = df.set_index(index_col)[feature_cols].copy()
    logger.info("Extracted %d exogenous feature(s): %s", len(feature_cols), feature_cols)
    return exog


def save_processed(df: pd.DataFrame, filepath: Optional[Path] = None) -> None:
    """
    Save the processed DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame to save.
    filepath : Optional[Path]
        Output path. Defaults to PROCESSED_DATA_PATH.
    """
    path = Path(filepath) if filepath else PROCESSED_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved processed data to '%s'", path)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_raw = load_raw_data()
    df_validated = validate_data(df_raw)
    ts = get_time_series(df_validated)
    train, test = train_test_split_ts(ts, test_size=15)

    print(f"\nTrain shape : {train.shape}")
    print(f"Test  shape : {test.shape}")
    print(f"\nFirst 3 train:\n{train.head(3)}")
    print(f"\nFirst 3 test:\n{test.head(3)}")