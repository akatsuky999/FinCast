from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / "Data" / "Datasets_return" / "manifest_fincast_price.yaml"
TARGET_COL = "target_close"
TIME_COL = "date"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = _resolve_path(path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    manifest["_path"] = str(manifest_path)
    return manifest


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").replace("\r", " ").split())


def _parse_json_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [_clean_text(v) for v in value if _clean_text(v)]
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [_clean_text(v) for v in parsed if _clean_text(v)]


def _series_floats(series: pd.Series) -> list[float]:
    return [_safe_float(v) for v in series.tolist()]


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    aliases: tuple[str, ...]
    full_csv: Path
    daily_news_csv: Path
    context_prompt_file: Path

    def matches(self, dataset_name: str) -> bool:
        token = dataset_name.strip()
        return token == self.name or token in self.aliases


class FinCastDataLoader:
    """Deterministic packet builder for FinCast price-forecasting windows."""

    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST_PATH):
        self.manifest_path = _resolve_path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        self.look_back = int(self.manifest.get("look_back", 60))
        self.predicted_window = int(self.manifest.get("predicted_window", 60))
        self.frequency = str(self.manifest.get("frequency", "B"))
        self.datasets = self._load_dataset_entries()

    def _load_dataset_entries(self) -> list[DatasetEntry]:
        entries: list[DatasetEntry] = []
        for raw in self.manifest.get("datasets", []) or []:
            aliases = tuple(str(a) for a in raw.get("aliases", []) or [])
            entries.append(
                DatasetEntry(
                    name=str(raw["name"]),
                    aliases=aliases,
                    full_csv=_resolve_path(raw["full_csv"]),
                    daily_news_csv=_resolve_path(raw["daily_news_csv"]),
                    context_prompt_file=_resolve_path(raw["context_prompt_file"]),
                )
            )
        return entries

    def resolve_dataset(self, dataset_name: str) -> DatasetEntry:
        for entry in self.datasets:
            if entry.matches(dataset_name):
                return entry
        known = ", ".join(entry.name for entry in self.datasets)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Known datasets: {known}")

    def _read_full(self, entry: DatasetEntry) -> pd.DataFrame:
        df = pd.read_csv(entry.full_csv)
        if TIME_COL not in df.columns:
            raise ValueError(f"{entry.full_csv} must contain '{TIME_COL}'")
        if TARGET_COL not in df.columns:
            raise ValueError(f"{entry.full_csv} must contain '{TARGET_COL}'")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)
        return df

    def _read_news(self, entry: DatasetEntry) -> pd.DataFrame:
        df = pd.read_csv(entry.daily_news_csv)
        if TIME_COL not in df.columns:
            raise ValueError(f"{entry.daily_news_csv} must contain '{TIME_COL}'")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)
        for col in ("news_count", "analyst_titles_json", "partner_headlines_json", "combined_news_text"):
            if col not in df.columns:
                df[col] = 0 if col == "news_count" else ""
        df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype(int)
        return df

    def _read_briefing(self, entry: DatasetEntry) -> str:
        try:
            return entry.context_prompt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def gather_forecast_inputs(
        self,
        dataset_name: str,
        window_offset: int = 0,
        forecast_horizon: int | None = None,
        look_back: int | None = None,
    ) -> dict[str, Any]:
        entry = self.resolve_dataset(dataset_name)
        full = self._read_full(entry)
        news = self._read_news(entry)

        L = int(look_back or self.look_back)
        H = int(forecast_horizon or self.predicted_window)
        offset = int(window_offset or 0)
        warnings: list[str] = []

        if offset < 0:
            warnings.append(f"Negative window_offset {offset} normalized to 0.")
            offset = 0
        if L <= 0 or H <= 0:
            raise ValueError("look_back and forecast_horizon must be positive integers.")

        lookback_start = offset
        lookback_end_exclusive = offset + L
        pred_end_exclusive = lookback_end_exclusive + H
        if pred_end_exclusive > len(full):
            raise ValueError(
                f"Window exceeds available data: offset={offset}, look_back={L}, "
                f"horizon={H}, rows={len(full)}"
            )

        lookback = full.iloc[lookback_start:lookback_end_exclusive].copy()
        pred_dates = full.iloc[lookback_end_exclusive:pred_end_exclusive][TIME_COL].copy()
        lookback_start_ts = pd.Timestamp(lookback[TIME_COL].iloc[0])
        lookback_end_ts = pd.Timestamp(lookback[TIME_COL].iloc[-1])
        prediction_start_ts = pd.Timestamp(pred_dates.iloc[0])

        news_window = news[(news[TIME_COL] >= lookback_start_ts) & (news[TIME_COL] <= lookback_end_ts)].copy()
        if news_window.empty:
            warnings.append("No aligned news was found inside this look-back window.")

        target_history = _series_floats(lookback[TARGET_COL])
        exog_cols = [c for c in lookback.columns if c not in {TIME_COL, TARGET_COL}]
        exogenous_history = {col: _series_floats(lookback[col]) for col in exog_cols}

        financial_features = self._financial_features(lookback, news_window)
        news_context = self._news_context(news_window, lookback[TIME_COL])

        ticker = entry.name.split("_")[-1]
        packet = {
            "dataset": entry.name,
            "ticker": ticker,
            "window_offset": offset,
            "look_back": L,
            "forecast_horizon": H,
            "look_back_start": lookback_start_ts.isoformat(),
            "look_back_end": lookback_end_ts.isoformat(),
            "look_back_timestamps": [pd.Timestamp(ts).isoformat() for ts in lookback[TIME_COL].tolist()],
            "prediction_start": prediction_start_ts.isoformat(),
            "prediction_timestamps": [pd.Timestamp(ts).isoformat() for ts in pred_dates.tolist()],
            "target_name": TARGET_COL,
            "target_history": target_history,
            "exogenous_history": exogenous_history,
            "financial_features": financial_features,
            "news_context": news_context,
            "dataset_briefing": self._read_briefing(entry),
            "leakage_policy": {
                "uses_future_target": False,
                "uses_future_news": False,
                "uses_future_exogenous": False,
                "description": "Only look-back window prices, lagged exogenous fields, and aligned look-back news are exposed.",
            },
            "warnings": warnings,
            "llm_summary": {
                "llm_summary_available": False,
                "news_summary": "",
                "financial_state_summary": "",
                "risk_notes": "",
            },
        }
        return json.loads(json.dumps(packet, default=_json_default, allow_nan=False))

    def _financial_features(self, lookback: pd.DataFrame, news_window: pd.DataFrame) -> dict[str, Any]:
        close = pd.to_numeric(lookback[TARGET_COL], errors="coerce").astype(float)
        log_close = np.log(close.to_numpy(dtype=float))
        returns = np.diff(log_close)
        if returns.size == 0:
            returns = np.array([0.0])

        running_max = np.maximum.accumulate(close.to_numpy(dtype=float))
        drawdown = close.to_numpy(dtype=float) / running_max - 1.0
        x = np.arange(len(log_close), dtype=float)
        trend_slope = float(np.polyfit(x, log_close, 1)[0]) if len(log_close) >= 2 else 0.0

        volume = pd.to_numeric(lookback.get("volume_lag1", pd.Series(dtype=float)), errors="coerce")
        if len(volume) >= 20:
            denom = _safe_float(volume.iloc[-20:].mean(), 1.0)
            volume_spike = _safe_float(volume.iloc[-1], 0.0) / denom if denom else 0.0
        else:
            volume_spike = 0.0

        news_counts = pd.to_numeric(news_window.get("news_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        return {
            "last_close": _safe_float(close.iloc[-1]),
            "first_close": _safe_float(close.iloc[0]),
            "cumulative_log_return": _safe_float(log_close[-1] - log_close[0]),
            "mean_daily_log_return": _safe_float(np.mean(returns)),
            "realized_volatility_daily": _safe_float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
            "realized_volatility_annualized": _safe_float(np.std(returns, ddof=1) * np.sqrt(252)) if returns.size > 1 else 0.0,
            "max_drawdown": _safe_float(np.min(drawdown)),
            "skewness": _safe_float(pd.Series(returns).skew()),
            "kurtosis": _safe_float(pd.Series(returns).kurt()),
            "trend_slope_log_price": _safe_float(trend_slope),
            "last_return_lag1": _safe_float(lookback["return_lag1"].iloc[-1]) if "return_lag1" in lookback else 0.0,
            "high_low_range_last": _safe_float(lookback["high_low_range_lag1"].iloc[-1]) if "high_low_range_lag1" in lookback else 0.0,
            "open_close_gap_last": _safe_float(lookback["open_close_gap_lag1"].iloc[-1]) if "open_close_gap_lag1" in lookback else 0.0,
            "volume_spike_ratio_20d": _safe_float(volume_spike),
            "news_total_lookback": int(news_counts.sum()) if len(news_counts) else 0,
            "news_active_days_lookback": int((news_counts > 0).sum()) if len(news_counts) else 0,
            "news_density_lookback": _safe_float((news_counts > 0).mean()) if len(news_counts) else 0.0,
        }

    def _news_context(self, news_window: pd.DataFrame, lookback_dates: pd.Series) -> dict[str, Any]:
        news_by_date = news_window.set_index(TIME_COL) if not news_window.empty else pd.DataFrame()
        trading_dates = [pd.Timestamp(ts).normalize() for ts in lookback_dates.tolist()]

        def bucket(days: int) -> dict[str, Any]:
            selected_dates = set(trading_dates[-days:])
            if news_window.empty:
                selected = news_window
            else:
                selected = news_window[news_window[TIME_COL].dt.normalize().isin(selected_dates)]
            headlines: list[str] = []
            for _, row in selected.sort_values(TIME_COL, ascending=False).iterrows():
                headlines.extend(_parse_json_list(row.get("analyst_titles_json")))
                headlines.extend(_parse_json_list(row.get("partner_headlines_json")))
            unique_headlines = []
            seen = set()
            for item in headlines:
                if item and item not in seen:
                    seen.add(item)
                    unique_headlines.append(item)
            combined = " | ".join(unique_headlines[:12])
            return {
                "trading_days": days,
                "news_count": int(selected["news_count"].sum()) if not selected.empty else 0,
                "active_news_days": int((selected["news_count"] > 0).sum()) if not selected.empty else 0,
                "top_headlines": unique_headlines[:10],
                "combined_text": combined[:4000],
            }

        all_headlines: list[str] = []
        if not news_window.empty:
            for _, row in news_window.sort_values(TIME_COL, ascending=False).iterrows():
                all_headlines.extend(_parse_json_list(row.get("analyst_titles_json")))
                all_headlines.extend(_parse_json_list(row.get("partner_headlines_json")))

        top_headlines = []
        seen = set()
        for headline in all_headlines:
            if headline and headline not in seen:
                seen.add(headline)
                top_headlines.append(headline)
            if len(top_headlines) >= 15:
                break

        total_count = int(pd.to_numeric(news_window.get("news_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        return {
            "recent_1d": bucket(1),
            "recent_3d": bucket(3),
            "recent_5d": bucket(5),
            "recent_20d": bucket(20),
            "top_headlines": top_headlines,
            "combined_recent_text": " | ".join(top_headlines[:15])[:5000],
            "news_density": _safe_float(total_count / max(len(trading_dates), 1)),
            "lookback_news_count": total_count,
            "latest_news_date": (
                pd.Timestamp(news_window[TIME_COL].max()).isoformat()
                if not news_window.empty
                else None
            ),
        }


def gather_forecast_inputs(
    dataset_name: str,
    window_offset: int = 0,
    forecast_horizon: int | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    look_back: int | None = None,
) -> dict[str, Any]:
    loader = FinCastDataLoader(manifest_path)
    return loader.gather_forecast_inputs(
        dataset_name=dataset_name,
        window_offset=window_offset,
        forecast_horizon=forecast_horizon,
        look_back=look_back,
    )


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "FinCastDataLoader",
    "gather_forecast_inputs",
    "load_manifest",
]
