"""
models.py
=========
Forecasting model classes for time series analysis.

Each class wraps a forecasting method behind a common interface:
    - fit(series)
    - predict(steps)
    - evaluate(y_true, y_pred)

This allows uniform benchmarking across naive, statistical, and
classical decomposition-based forecasting approaches.

Author: [Your Name]
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Actual observed values.
    y_pred : np.ndarray
        Forecasted values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'MAE', 'RMSE', 'MAPE'.

    Raises
    ------
    ValueError
        If `y_true` and `y_pred` have different lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("'y_true' and 'y_pred' must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"'y_true' (len={len(y_true)}) and 'y_pred' (len={len(y_pred)}) "
            "must have the same length"
        )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Avoid division by zero in MAPE by masking zero actuals
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() == 0:
        mape = np.nan
        logger.warning("All y_true values are zero; MAPE is undefined (NaN)")
    else:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        )

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


class BaseForecaster:
    """
    Base class defining the common forecaster interface.

    Subclasses must implement `fit` and `predict`. `evaluate` is
    provided here and shared by all subclasses.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._fitted = False
        self._history: Optional[pd.Series] = None

    def fit(self, series: pd.Series) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, steps: int) -> pd.Series:
        raise NotImplementedError

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using MAE, RMSE, and MAPE.

        Parameters
        ----------
        y_true : pd.Series
            Ground-truth values for the forecast horizon.
        y_pred : pd.Series
            Forecasted values for the same horizon.

        Returns
        -------
        Dict[str, float]
            Dictionary with 'MAE', 'RMSE', 'MAPE' keys.
        """
        metrics = _evaluate_metrics(y_true.values, y_pred.values)
        logger.info(
            "[%s] Evaluation -> MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%",
            self.name, metrics["MAE"], metrics["RMSE"], metrics["MAPE"],
        )
        return metrics

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} model has not been fitted. Call .fit() first.")

    def _future_index(self, steps: int) -> pd.DatetimeIndex:
        """Generate a future DatetimeIndex continuing from the training series."""
        if not isinstance(self._history.index, pd.DatetimeIndex):
            return pd.RangeIndex(start=len(self._history), stop=len(self._history) + steps)
        freq = self._history.index.freq or pd.infer_freq(self._history.index) or "D"
        return pd.date_range(
            start=self._history.index[-1], periods=steps + 1, freq=freq
        )[1:]


class NaiveForecaster(BaseForecaster):
    """
    Naive forecast: repeats the last observed value for all future steps.

    Useful as the simplest possible benchmark — any more complex model
    should outperform this to justify its added complexity.
    """

    def __init__(self) -> None:
        super().__init__(name="Naive")
        self._last_value: Optional[float] = None

    def fit(self, series: pd.Series) -> "NaiveForecaster":
        """
        Fit the naive model by storing the last observed value.

        Parameters
        ----------
        series : pd.Series
            Training time series.

        Returns
        -------
        NaiveForecaster
            Self, for method chaining.

        Raises
        ------
        ValueError
            If `series` is empty.
        """
        if series.empty:
            raise ValueError("'series' must not be empty")

        self._history = series
        self._last_value = float(series.iloc[-1])
        self._fitted = True
        logger.info("[%s] Fitted with last value = %.4f", self.name, self._last_value)
        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Forecast `steps` periods ahead by repeating the last value.

        Parameters
        ----------
        steps : int
            Number of future periods to forecast.

        Returns
        -------
        pd.Series
            Forecasted values indexed by future dates.

        Raises
        ------
        RuntimeError
            If called before `fit`.
        ValueError
            If `steps` is not a positive integer.
        """
        self._check_fitted()
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"'steps' must be a positive integer, got {steps}")

        idx = self._future_index(steps)
        return pd.Series([self._last_value] * steps, index=idx, name="naive_forecast")


class MovingAverageForecaster(BaseForecaster):
    """
    Moving average forecast: forecasts all future steps as the mean
    of the last `window` observed values.
    """

    def __init__(self, window: int = 7) -> None:
        super().__init__(name="MovingAverage")
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"'window' must be a positive integer, got {window}")
        self.window = window
        self._ma_value: Optional[float] = None

    def fit(self, series: pd.Series) -> "MovingAverageForecaster":
        """
        Fit by computing the mean of the last `window` observations.

        Parameters
        ----------
        series : pd.Series
            Training time series.

        Returns
        -------
        MovingAverageForecaster
            Self, for method chaining.

        Raises
        ------
        ValueError
            If `series` has fewer observations than `window`.
        """
        if series.empty:
            raise ValueError("'series' must not be empty")
        if len(series) < self.window:
            raise ValueError(
                f"'series' length ({len(series)}) must be >= window ({self.window})"
            )

        self._history = series
        self._ma_value = float(series.tail(self.window).mean())
        self._fitted = True
        logger.info("[%s] Fitted with %d-period mean = %.4f", self.name, self.window, self._ma_value)
        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Forecast `steps` periods ahead using the fitted moving average.

        Parameters
        ----------
        steps : int
            Number of future periods to forecast.

        Returns
        -------
        pd.Series
            Forecasted values indexed by future dates.
        """
        self._check_fitted()
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"'steps' must be a positive integer, got {steps}")

        idx = self._future_index(steps)
        return pd.Series([self._ma_value] * steps, index=idx, name="ma_forecast")


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Simple Exponential Smoothing (SES) forecaster.

    Suitable for series with no clear trend or seasonality; weights
    recent observations more heavily via a smoothing parameter alpha.
    """

    def __init__(self, smoothing_level: Optional[float] = None) -> None:
        super().__init__(name="ExponentialSmoothing")
        if smoothing_level is not None and not (0 < smoothing_level <= 1):
            raise ValueError("'smoothing_level' must be in (0, 1] if provided")
        self.smoothing_level = smoothing_level
        self._model_fit = None

    def fit(self, series: pd.Series) -> "ExponentialSmoothingForecaster":
        """
        Fit a SimpleExpSmoothing model.

        Parameters
        ----------
        series : pd.Series
            Training time series.

        Returns
        -------
        ExponentialSmoothingForecaster
            Self, for method chaining.

        Raises
        ------
        ValueError
            If `series` is empty.
        RuntimeError
            If the underlying statsmodels fit fails.
        """
        if series.empty:
            raise ValueError("'series' must not be empty")

        self._history = series
        try:
            model = SimpleExpSmoothing(series, initialization_method="estimated")
            if self.smoothing_level is not None:
                self._model_fit = model.fit(smoothing_level=self.smoothing_level, optimized=False)
            else:
                self._model_fit = model.fit()
        except Exception as exc:
            raise RuntimeError(f"SimpleExpSmoothing fit failed: {exc}") from exc

        self._fitted = True
        logger.info("[%s] Fitted. alpha=%.4f", self.name, self._model_fit.params.get("smoothing_level", np.nan))
        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Forecast `steps` periods ahead.

        Parameters
        ----------
        steps : int
            Number of future periods to forecast.

        Returns
        -------
        pd.Series
            Forecasted values indexed by future dates.
        """
        self._check_fitted()
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"'steps' must be a positive integer, got {steps}")

        forecast = self._model_fit.forecast(steps)
        idx = self._future_index(steps)
        return pd.Series(forecast.values, index=idx, name="ses_forecast")


class HoltWintersForecaster(BaseForecaster):
    """
    Holt-Winters forecaster with optional trend and seasonal components.
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: Optional[int] = 7,
    ) -> None:
        super().__init__(name="HoltWinters")
        valid_components = {None, "add", "mul"}
        if trend not in valid_components or seasonal not in valid_components:
            raise ValueError("'trend' and 'seasonal' must be one of None, 'add', 'mul'")
        if seasonal is not None and (seasonal_periods is None or seasonal_periods <= 1):
            raise ValueError("'seasonal_periods' must be > 1 when 'seasonal' is set")

        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._model_fit = None

    def fit(self, series: pd.Series) -> "HoltWintersForecaster":
        """
        Fit a Holt-Winters Exponential Smoothing model.

        Parameters
        ----------
        series : pd.Series
            Training time series.

        Returns
        -------
        HoltWintersForecaster
            Self, for method chaining.

        Raises
        ------
        ValueError
            If `series` is too short for the requested seasonal period.
        RuntimeError
            If the underlying statsmodels fit fails.
        """
        if series.empty:
            raise ValueError("'series' must not be empty")
        if self.seasonal is not None and len(series) < 2 * self.seasonal_periods:
            raise ValueError(
                f"'series' must have at least 2 * seasonal_periods "
                f"({2 * self.seasonal_periods}) observations, got {len(series)}"
            )

        self._history = series
        try:
            self._model_fit = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            ).fit()
        except Exception as exc:
            raise RuntimeError(f"Holt-Winters fit failed: {exc}") from exc

        self._fitted = True
        logger.info("[%s] Fitted with trend=%s, seasonal=%s, periods=%s",
                     self.name, self.trend, self.seasonal, self.seasonal_periods)
        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Forecast `steps` periods ahead.

        Parameters
        ----------
        steps : int
            Number of future periods to forecast.

        Returns
        -------
        pd.Series
            Forecasted values indexed by future dates.
        """
        self._check_fitted()
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"'steps' must be a positive integer, got {steps}")

        forecast = self._model_fit.forecast(steps)
        idx = self._future_index(steps)
        return pd.Series(forecast.values, index=idx, name="holtwinters_forecast")


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA(p, d, q) forecaster.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)) -> None:
        super().__init__(name="ARIMA")
        if len(order) != 3 or any(not isinstance(x, int) or x < 0 for x in order):
            raise ValueError(f"'order' must be a tuple of 3 non-negative ints, got {order}")
        self.order = order
        self._model_fit = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """
        Fit an ARIMA model with the configured (p, d, q) order.

        Parameters
        ----------
        series : pd.Series
            Training time series.

        Returns
        -------
        ARIMAForecaster
            Self, for method chaining.

        Raises
        ------
        ValueError
            If `series` is empty.
        RuntimeError
            If the underlying statsmodels fit fails (e.g., convergence issues).
        """
        if series.empty:
            raise ValueError("'series' must not be empty")

        self._history = series
        try:
            self._model_fit = ARIMA(series, order=self.order).fit()
        except Exception as exc:
            raise RuntimeError(f"ARIMA{self.order} fit failed: {exc}") from exc

        self._fitted = True
        logger.info("[%s] Fitted ARIMA%s | AIC=%.2f", self.name, self.order, self._model_fit.aic)
        return self

    def predict(self, steps: int, return_conf_int: bool = False, alpha: float = 0.05):
        """
        Forecast `steps` periods ahead, optionally with confidence intervals.

        Parameters
        ----------
        steps : int
            Number of future periods to forecast.
        return_conf_int : bool
            If True, also return a DataFrame of confidence intervals.
        alpha : float
            Significance level for confidence intervals (default 0.05 -> 95% CI).

        Returns
        -------
        pd.Series or Tuple[pd.Series, pd.DataFrame]
            Forecasted values, and optionally confidence intervals.
        """
        self._check_fitted()
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"'steps' must be a positive integer, got {steps}")

        forecast_obj = self._model_fit.get_forecast(steps=steps)
        idx = self._future_index(steps)
        mean_forecast = pd.Series(forecast_obj.predicted_mean.values, index=idx, name="arima_forecast")

        if return_conf_int:
            conf_int = forecast_obj.conf_int(alpha=alpha)
            conf_int.index = idx
            return mean_forecast, conf_int
        return mean_forecast


class SARIMAXForecaster(BaseForecaster):
    """
    SARIMAX forecaster supporting seasonal terms and exogenous regressors.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        super().__init__(name="SARIMAX")
        if len(order) != 3:
            raise ValueError(f"'order' must be a 3-tuple, got {order}")
        if len(seasonal_order) != 4:
            raise ValueError(f"'seasonal_order' must be a 4-tuple, got {seasonal_order}")

        self.order = order
        self.seasonal_order = seasonal_order
        self._model_fit = None
        self._exog_columns: Optional[list] = None

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "SARIMAXForecaster":
        """
        Fit a SARIMAX model, optionally with exogenous regressors.

        Parameters
        ----------
        series : pd.Series
            Training time series (endogenous variable).
        exog : Optional[pd.DataFrame]
            Exogenous regressors aligned with `series` by index.

        Returns
        -------
        SARIMAXForecaster
            Self, for method chaining.

        Raises
        ------
        ValueError
            If `series` is empty, or `exog` index does not align with `series`.
        RuntimeError
            If the underlying statsmodels fit fails.
        """
        if series.empty:
            raise ValueError("'series' must not be empty")
        if exog is not None and not exog.index.equals(series.index):
            raise ValueError("'exog' index must exactly match 'series' index")

        self._history = series
        self._exog_columns = list(exog.columns) if exog is not None else None

        try:
            self._model_fit = SARIMAX(
                series,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        except Exception as exc:
            raise RuntimeError(f"SARIMAX{self.order}{self.seasonal_order} fit failed: {exc}") from exc

        self._fitted = True
        logger.info(
            "[%s] Fitted SARIMAX%s%s | exog=%s | AIC=%.2f",
            self.name, self.order, self.seasonal_order, self._exog_columns, self._model_fit.aic,
        )
        return self

    def predict(
        self,
        steps: int,
        exog_future: Optional[pd.DataFrame] = None,
        return_conf_int: bool = False,
        alpha: float = 0.05,
    ):
        """
        Forecast `steps` periods ahead, optionally with exogenous future
        values and confidence intervals.

        Parameters
        ----------
        steps : int
            Number of future periods to forecast.
        exog_future : Optional[pd.DataFrame]
            Future values of exogenous regressors. Required if the model
            was fit with `exog`.
        return_conf_int : bool
            If True, also return a DataFrame of confidence intervals.
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        pd.Series or Tuple[pd.Series, pd.DataFrame]
            Forecasted values, and optionally confidence intervals.

        Raises
        ------
        ValueError
            If the model requires exogenous data but `exog_future` is
            missing or has the wrong shape.
        """
        self._check_fitted()
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"'steps' must be a positive integer, got {steps}")

        if self._exog_columns is not None:
            if exog_future is None:
                raise ValueError("Model was fit with exogenous regressors; 'exog_future' is required")
            if len(exog_future) != steps:
                raise ValueError(
                    f"'exog_future' must have {steps} rows, got {len(exog_future)}"
                )
            if list(exog_future.columns) != self._exog_columns:
                raise ValueError(
                    f"'exog_future' columns {list(exog_future.columns)} "
                    f"do not match fitted exog columns {self._exog_columns}"
                )

        forecast_obj = self._model_fit.get_forecast(steps=steps, exog=exog_future)
        idx = self._future_index(steps)
        mean_forecast = pd.Series(forecast_obj.predicted_mean.values, index=idx, name="sarimax_forecast")

        if return_conf_int:
            conf_int = forecast_obj.conf_int(alpha=alpha)
            conf_int.index = idx
            return mean_forecast, conf_int
        return mean_forecast


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a small synthetic series with a level shift
    rng = pd.date_range("2026-01-01", periods=60, freq="D")
    values = np.concatenate([
        100 + np.random.normal(0, 3, 30),
        6 + np.random.normal(0, 2, 30),
    ])
    series = pd.Series(values, index=rng, name="daily_ship_transits")

    train, test = series.iloc[:-7], series.iloc[-7:]

    print("\n=== Naive ===")
    naive = NaiveForecaster().fit(train)
    naive_pred = naive.predict(steps=7)
    naive.evaluate(test, naive_pred)

    print("\n=== Moving Average ===")
    ma = MovingAverageForecaster(window=7).fit(train)
    ma_pred = ma.predict(steps=7)
    ma.evaluate(test, ma_pred)

    print("\n=== Exponential Smoothing ===")
    ses = ExponentialSmoothingForecaster().fit(train)
    ses_pred = ses.predict(steps=7)
    ses.evaluate(test, ses_pred)

    print("\n=== Holt-Winters ===")
    hw = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=7).fit(train)
    hw_pred = hw.predict(steps=7)
    hw.evaluate(test, hw_pred)

    print("\n=== ARIMA ===")
    arima = ARIMAForecaster(order=(1, 1, 1)).fit(train)
    arima_pred = arima.predict(steps=7)
    arima.evaluate(test, arima_pred)

    print("\n=== SARIMAX (no exog) ===")
    sarimax = SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 7)).fit(train)
    sarimax_pred = sarimax.predict(steps=7)
    sarimax.evaluate(test, sarimax_pred)
