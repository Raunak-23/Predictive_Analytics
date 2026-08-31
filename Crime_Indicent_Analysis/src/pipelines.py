import pandas as pd

def build_series(df, date_col, location_col, location_value, freq,
                 category_col=None, category_value=None):
    """
    Filter incident-level records to a specific location (and optional category)
    and aggregate to a regular time series.
    """
    loc_df = df.copy()
    loc_df = loc_df[loc_df[location_col].astype(str) == str(location_value)]
    if category_col is not None and category_value is not None:
        loc_df = loc_df[loc_df[category_col].astype(str).str.upper() == category_value.upper()]
    if len(loc_df) == 0:
        raise ValueError(f"No records found for filters.")
    series = (
        loc_df.set_index(date_col)
        .resample(freq)
        .size()
        .asfreq(freq, fill_value=0)
    )
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    if not series.index.is_unique:
        series = series[~series.index.duplicated(keep='first')]
    series.name = 'incidents'
    return series

# --- BaseForecastPipeline and NaivePipeline ---
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseForecastPipeline(ABC):
    """Abstract base class for all forecasting pipelines."""

    @property
    @abstractmethod
    def name(self):
        """Human-readable name of the model."""
        pass

    @abstractmethod
    def fit(self, train_series: pd.Series):
        """
        Fit the model on the training series (in-sample period).
        Must store any learned parameters internally.
        """
        pass

    @abstractmethod
    def predict(self, n_periods: int) -> np.ndarray:
        """
        Generate forecasts for the next `n_periods` steps beyond the training end.
        Returns a numpy array of length n_periods.
        """
        pass

class NaivePipeline(BaseForecastPipeline):
    """Naive (persistence) forecast: repeats the last observed value."""

    @property
    def name(self):
        return "Naive (persistence)"

    def fit(self, train_series: pd.Series):
        # Store the last value of the training series
        self.last_value_ = float(train_series.iloc[-1])
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        return np.full(n_periods, self.last_value_)

# --- ARPipeline ---
from statsmodels.tsa.ar_model import AutoReg

class ARPipeline(BaseForecastPipeline):
    """
    Autoregressive model (AR) with a specified lag order.
    Uses trend='ct' (constant + time trend) by default, as per the manual.
    """
    def __init__(self, lags: int, trend: str = 'ct'):
        self.lags = lags
        self.trend = trend
        self._model = None
        self._fit_result = None
        self._train_end = None

    @property
    def name(self):
        return f"AR({self.lags})"

    def fit(self, train_series: pd.Series):
        self._train_end = train_series.index.max()
        self._model = AutoReg(
            train_series,
            lags=self.lags,
            trend=self.trend
        )
        self._fit_result = self._model.fit()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        # Predict from the end of the training series, 'dynamic=False' for one-step-ahead
        # using actual lagged values from the training data for the first forecast.
        pred = self._fit_result.predict(
            start=len(self._fit_result.data.endog),
            end=len(self._fit_result.data.endog) + n_periods - 1,
            dynamic=False
        )
        return np.asarray(pred)

    def get_coefficients(self):
        """Return the fitted AR coefficients and standard errors for interpretation."""
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted.")
        return self._fit_result.params

    def get_summary(self):
        """Return the full summary table from the fitted model."""
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted.")
        return self._fit_result.summary()

# --- ARIMAPipeline ---
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import json
import pandas as pd
from checkpointing import run_resumable

class ARIMAPipeline(BaseForecastPipeline):
    """
    ARIMA model with given order (p, d, q).
    The 'backend' parameter is provided for later batched GPU usage (Stage 14d);
    in this stage we always use 'cpu'.
    """
    def __init__(self, order, trend, backend='cpu'):
        self.order = tuple(order)
        self.trend = trend
        self.backend = backend
        self._model = None
        self._fit_result = None

    @property
    def name(self):
        return f"ARIMA{self.order}"

    def fit(self, train_series: pd.Series):
        # For this core stage, ignore backend; always use statsmodels CPU implementation
        self._model = ARIMA(train_series, order=self.order, trend=self.trend)
        self._fit_result = self._model.fit()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        pred = self._fit_result.forecast(steps=n_periods)
        return np.asarray(pred)

    def get_aic(self):
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted.")
        return self._fit_result.aic

    def get_summary(self):
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted.")
        return self._fit_result.summary()

    def predict_with_interval(self, n_periods: int, alpha: float = 0.05):
        """
        Return point forecasts and prediction-interval bounds.

        Parameters
        ----------
        n_periods : int
            Number of steps to forecast beyond the training end.
        alpha : float
            Significance level; 0.05 gives 95 % intervals.

        Returns
        -------
        tuple : (pred_mean, lower, upper) -- each a np.ndarray of length n_periods
        """
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        fc = self._fit_result.get_forecast(steps=n_periods)
        pred_mean = np.asarray(fc.predicted_mean)
        ci        = fc.conf_int(alpha=alpha)
        lower     = np.asarray(ci.iloc[:, 0])
        upper     = np.asarray(ci.iloc[:, 1])
        return pred_mean, lower, upper


# --- Generic order selection with checkpointing ---

def select_best_order(train_series, validation_series, candidate_orders, trend_set,
                      pipeline_class, pipeline_kwargs_fn, results_path=None):
    """
    Fit ARIMA models for each candidate order and trend.
    Models are ranked primarily by validation MAE, followed by validation RMSE
    and training AIC as tie-breakers.
    If results_path is provided, uses checkpointing to resume from previous runs.

    Parameters
    ----------
    train_series : pd.Series
        Training data.
    candidate_orders : list of tuples
        List of (p, d, q) orders to evaluate.
    trend_set : tuple of str
        List of trend specifications to evaluate (e.g., ['n', 'c', 't', 'ct']).
    pipeline_class : class
        Must be a subclass of BaseForecastPipeline with an __init__ that accepts 'order'.
    pipeline_kwargs_fn : callable
        Function that takes an order and returns a dict of additional kwargs for the pipeline.
    results_path : str or Path, optional
        Path to .jsonl file for checkpointing. If None, no checkpointing is used (but still
        loops sequentially without resumability).

    Returns
    -------
    pd.DataFrame
        Sorted by validation MAE, validation RMSE, and training AIC
    """
def select_best_order(train_series, validation_series, candidate_orders, trend_set,
                      pipeline_class, pipeline_kwargs_fn, results_path=None):

    grid = []

    for order in candidate_orders:
        order = tuple(order)
        d = order[1]

        if d == 0:
            valid_trends = trend_set
        elif d == 1:
            valid_trends = trend_set[:2]  # only 'n' and 'c' are valid for d=1
        else:
            valid_trends = trend_set[:1]  # only 'n' is valid for d>1

        for trend in valid_trends:
            grid.append({
                'order': order,
                'trend': trend
            })

    def key_fn(config):
        return f"{config['order']}|trend={config['trend']}"

    def process_config(config):
        order = tuple(config['order'])
        trend = config['trend']

        try:
            kwargs = (
                pipeline_kwargs_fn(order, trend)
                if pipeline_kwargs_fn
                else {}
            )

            model = pipeline_class(
                order=order,
                trend=trend,
                **kwargs
            )

            model.fit(train_series)
            validation_pred = model.predict(len(validation_series))

            actual = np.asarray(validation_series)
            predicted = np.asarray(validation_pred)

            validation_mae = float(
                np.mean(np.abs(actual - predicted))
            )

            validation_rmse = float(
                np.sqrt(np.mean((actual - predicted) ** 2))
            )

            return {
                'order': list(order),
                'trend': trend,
                'aic': model.get_aic(),
                'validation_mae': validation_mae,
                'validation_rmse': validation_rmse
            }

        except Exception as error:
            print(f"Failed {order}, trend={trend}: {error}")

            return {
                'order': list(order),
                'trend': trend,
                'aic': None,
                'validation_mae': None,
                'validation_rmse': None
            }

    if results_path is not None:
        results_path = Path(results_path)

        run_resumable(
            items=grid,
            key_fn=key_fn,
            process_fn=process_config,
            results_path=results_path,
            desc='ARIMA order/trend grid search'
        )

        completed = []

        if results_path.exists():
            with open(results_path, 'r') as file:
                for line in file:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        if (
                            data.get('validation_mae') is not None
                            and data.get('validation_rmse') is not None
                        ):
                            completed.append({
                                'order': data['order'],
                                'trend': data['trend'],
                                'aic': data['aic'],
                                'validation_mae': data['validation_mae'],
                                'validation_rmse': data['validation_rmse']
                            })

                    except json.JSONDecodeError:
                        continue
    else:
        completed = []

        for config in grid:
            result = process_config(config)

            if result['validation_mae'] is not None:
                completed.append(result)

    if not completed:
        raise RuntimeError(
            "No ARIMA order/trend combination fitted successfully."
        )

    ranked = pd.DataFrame(completed)

    # Primary selection criterion: validation MAE.
    # Tie breakers: validation RMSE, then training AIC.
    ranked = ranked.sort_values(
        by=[
            'validation_mae',
            'validation_rmse',
            'aic'
        ],
        ascending=True
    ).reset_index(drop=True)

    return ranked

# ------------------------------------------------------------------
# 14a — SARIMAPipeline
# ------------------------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAPipeline(BaseForecastPipeline):
    """
    Seasonal ARIMA wrapper using statsmodels SARIMAX.
    """
    def __init__(self, order, seasonal_order, trend='n'):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self._model = None
        self._fit_result = None

    @property
    def name(self):
        return f"SARIMA{self.order}x{self.seasonal_order}"

    def fit(self, train_series: pd.Series):
        self._model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self._fit_result = self._model.fit(disp=False)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        pred = self._fit_result.forecast(steps=n_periods)
        return np.asarray(pred)

    def predict_with_interval(self, n_periods: int, alpha: float = 0.05):
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted.")
        fc = self._fit_result.get_forecast(steps=n_periods)
        pred_mean = np.asarray(fc.predicted_mean)
        ci = fc.conf_int(alpha=alpha)
        lower = np.asarray(ci.iloc[:, 0])
        upper = np.asarray(ci.iloc[:, 1])
        return pred_mean, lower, upper

    def get_aic(self):
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted.")
        return self._fit_result.aic