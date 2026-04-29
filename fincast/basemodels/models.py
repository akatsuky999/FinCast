from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COL = "target_close"
TIME_COL = "date"
BASELINE_VERSION = "price_baselines_v2"


@dataclass
class BaselineForecast:
    model_name: str
    predictions: list[float]
    metadata: dict
    warning: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class BasePriceModel:
    name = "BasePriceModel"

    def fit_predict(
        self,
        window_df: pd.DataFrame,
        horizon: int,
        prediction_timestamps: Iterable[str] | None = None,
    ) -> BaselineForecast:
        raise NotImplementedError


def _close(window_df: pd.DataFrame) -> np.ndarray:
    arr = pd.to_numeric(window_df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("target_close history must be non-empty and finite")
    return arr


def _log_returns(close: np.ndarray) -> np.ndarray:
    if close.size < 2:
        return np.array([0.0], dtype=float)
    return np.diff(np.log(np.maximum(close, 1e-12)))


def _positive(values: np.ndarray, fallback: float) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, fallback)
    arr = np.maximum(arr, 1e-6)
    return [float(v) for v in arr.tolist()]


def _price_from_returns(last_close: float, returns: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    return float(last_close) * np.exp(np.cumsum(returns))


def _price_from_log_forecast(log_values: np.ndarray, fallback: float) -> np.ndarray:
    fallback = max(float(fallback), 1e-6)
    lower = np.log(fallback * 0.50)
    upper = np.log(fallback * 2.0)
    return np.exp(np.clip(np.asarray(log_values, dtype=float), lower, upper))


def _repeat(last_close: float, horizon: int) -> list[float]:
    return [float(last_close)] * int(horizon)


def _safe_forecast(model: BasePriceModel, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
    last_close = float(_close(window_df)[-1])
    try:
        forecast = model.fit_predict(window_df, horizon, prediction_timestamps)
        if len(forecast.predictions) != horizon:
            raise ValueError(f"{model.name} produced {len(forecast.predictions)} values, expected {horizon}")
        preds = np.asarray(forecast.predictions, dtype=float)
        if not np.isfinite(preds).all() or np.any(preds <= 0):
            raise ValueError(f"{model.name} produced non-finite or non-positive prices")
        ratio = preds / max(last_close, 1e-6)
        if np.any(ratio < 0.50) or np.any(ratio > 2.0):
            raise ValueError(f"{model.name} produced implausible price scale relative to last_close")
        return forecast
    except Exception as exc:
        return BaselineForecast(
            model_name=model.name,
            predictions=_repeat(last_close, horizon),
            metadata={"fallback": "LastClose"},
            warning=str(exc),
        )


class LastCloseModel(BasePriceModel):
    name = "LastClose"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        last = float(_close(window_df)[-1])
        return BaselineForecast(self.name, _repeat(last, horizon), {"last_close": last})


class RandomWalkDriftModel(BasePriceModel):
    name = "RandomWalkDrift"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        close = _close(window_df)
        r = _log_returns(close)
        drift = float(np.nanmean(r)) if r.size else 0.0
        steps = np.arange(1, horizon + 1, dtype=float)
        pred = close[-1] * np.exp(drift * steps)
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"drift": drift})


class EWMAReturnModel(BasePriceModel):
    name = "EWMAReturn"

    def __init__(self, span: int = 20):
        self.span = span

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        close = _close(window_df)
        r = pd.Series(_log_returns(close))
        ewma = float(r.ewm(span=self.span, adjust=False).mean().iloc[-1]) if len(r) else 0.0
        pred = _price_from_returns(close[-1], np.full(horizon, ewma))
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"span": self.span, "ewma_return": ewma})


class ARReturnModel(BasePriceModel):
    name = "ARReturn"

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        from statsmodels.tsa.ar_model import AutoReg

        close = _close(window_df)
        r = _log_returns(close)
        lag = max(1, min(self.max_lag, len(r) // 4))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = AutoReg(r, lags=lag, old_names=False).fit()
            pred_r = np.asarray(fit.forecast(horizon), dtype=float)
        pred = _price_from_returns(close[-1], pred_r)
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"lag": lag})


class ARMAReturnModel(BasePriceModel):
    name = "ARMAReturn"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        from statsmodels.tsa.arima.model import ARIMA

        close = _close(window_df)
        r = _log_returns(close)
        best = None
        best_aic = float("inf")
        best_order = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in ((1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (1, 0, 2)):
                try:
                    fit = ARIMA(r, order=order, trend="c").fit()
                    if float(fit.aic) < best_aic:
                        best, best_aic, best_order = fit, float(fit.aic), order
                except Exception:
                    continue
            if best is None:
                raise RuntimeError("No ARMA order converged")
            pred_r = np.asarray(best.forecast(horizon), dtype=float)
        pred = _price_from_returns(close[-1], pred_r)
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"order": best_order, "aic": best_aic})


class ARIMALogPriceModel(BasePriceModel):
    name = "ARIMALogPrice"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        from statsmodels.tsa.arima.model import ARIMA

        close = _close(window_df)
        y = np.log(close)
        best = None
        best_aic = float("inf")
        best_order = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in ((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1)):
                try:
                    fit = ARIMA(y, order=order, trend="n").fit()
                    if float(fit.aic) < best_aic:
                        best, best_aic, best_order = fit, float(fit.aic), order
                except Exception:
                    continue
            if best is None:
                raise RuntimeError("No ARIMA log-price order converged")
            pred_log = np.asarray(best.forecast(horizon), dtype=float)
        return BaselineForecast(self.name, _positive(_price_from_log_forecast(pred_log, close[-1]), close[-1]), {"order": best_order, "aic": best_aic})


class ARIMAXPriceModel(BasePriceModel):
    name = "ARIMAXPrice"

    exog_cols = [
        "return_lag1",
        "high_low_range_lag1",
        "open_close_gap_lag1",
        "log_volume_lag1",
        "news_count_lag1",
    ]

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        close = _close(window_df)
        y = np.log(close)
        available = [c for c in self.exog_cols if c in window_df.columns]
        if not available:
            raise RuntimeError("No exogenous columns available")
        exog = window_df[available].apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy(dtype=float)
        future_exog = np.repeat(exog[-1:, :], horizon, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(
                y,
                exog=exog,
                order=(1, 1, 1),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=80)
            pred_log = np.asarray(fit.forecast(horizon, exog=future_exog), dtype=float)
        return BaselineForecast(self.name, _positive(_price_from_log_forecast(pred_log, close[-1]), close[-1]), {"order": (1, 1, 1), "exog_cols": available})


class ARMAGARCHReturnModel(BasePriceModel):
    name = "ARMAGARCHReturn"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        from arch import arch_model

        close = _close(window_df)
        r = _log_returns(close) * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(r, mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist="t", rescale=False)
            fit = model.fit(disp="off", show_warning=False)
            fcst = fit.forecast(horizon=horizon, reindex=False)
        mean = np.asarray(fcst.mean.iloc[-1].to_numpy(dtype=float), dtype=float) / 100.0
        variance = np.asarray(fcst.variance.iloc[-1].to_numpy(dtype=float), dtype=float) / (100.0**2)
        pred = _price_from_returns(close[-1], mean)
        return BaselineForecast(
            self.name,
            _positive(pred, close[-1]),
            {"mean_model": "ARX(1)", "vol_model": "GARCH(1,1)", "variance_forecast": variance.tolist()},
        )


class SeasonalNaiveModel(BasePriceModel):
    name = "SeasonalNaive"

    def __init__(self, season_length: int = 5):
        self.season_length = season_length

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        close = _close(window_df)
        s = max(1, min(self.season_length, len(close)))
        pattern = close[-s:]
        reps = int(np.ceil(horizon / s))
        pred = np.tile(pattern, reps)[:horizon]
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"season_length": s})


class HistoricAverageModel(BasePriceModel):
    name = "HistoricAverage"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        close = _close(window_df)
        mean_val = float(np.mean(close))
        pred = np.full(horizon, mean_val, dtype=float)
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"mean": mean_val})


class AutoETSModel(BasePriceModel):
    name = "AutoETS"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS as SFAutoETS

        close = _close(window_df)
        dates = pd.to_datetime(window_df["date"])
        freq = pd.infer_freq(dates) or "D"
        df = pd.DataFrame({"unique_id": "series", "ds": dates, "y": close})
        sf = StatsForecast(models=[SFAutoETS(season_length=5)], freq=freq)
        sf.fit(df)
        fcst_df = sf.predict(h=horizon)
        pred = np.asarray(fcst_df["AutoETS"], dtype=float)
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"season_length": 5})


class ThetaModel(BasePriceModel):
    name = "Theta"

    def fit_predict(self, window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> BaselineForecast:
        from statsmodels.tsa.forecasting.theta import ThetaModel as StatsmodelsTheta

        close = _close(window_df)
        fit = StatsmodelsTheta(close, period=5, use_test=True, method="auto").fit()
        pred = np.asarray(fit.forecast(steps=horizon), dtype=float)
        return BaselineForecast(self.name, _positive(pred, close[-1]), {"period": 5})


def get_baseline_models() -> list[BasePriceModel]:
    return [
        LastCloseModel(),
        RandomWalkDriftModel(),
        EWMAReturnModel(),
        ARReturnModel(),
        ARMAReturnModel(),
        ARIMALogPriceModel(),
        ARIMAXPriceModel(),
        ARMAGARCHReturnModel(),
        SeasonalNaiveModel(),
        HistoricAverageModel(),
        AutoETSModel(),
        ThetaModel(),
    ]


def run_all_baselines(window_df: pd.DataFrame, horizon: int, prediction_timestamps=None) -> dict[str, dict]:
    forecasts = {}
    for model in get_baseline_models():
        forecast = _safe_forecast(model, window_df, horizon, prediction_timestamps)
        forecasts[model.name] = forecast.to_dict()
    return forecasts
