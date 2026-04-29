from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_default(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_default(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [json_default(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Path):
        return str(value)
    return value


def parse_jsonish(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = str(text or "").strip()
    if "```" in raw:
        for part in raw.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                raw = candidate
                break
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except Exception:
        return dict(fallback or {})
    return parsed if isinstance(parsed, dict) else dict(fallback or {})


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def finite_array(values: Iterable[Any], fallback: float | None = None) -> np.ndarray:
    arr = np.asarray([safe_float(v, np.nan) for v in values], dtype=float)
    if fallback is not None:
        arr = np.where(np.isfinite(arr), arr, float(fallback))
    return arr


def mae(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    truth, pred = _aligned_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred - truth))) if truth.size else float("nan")


def rmse(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    truth, pred = _aligned_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - truth) ** 2))) if truth.size else float("nan")


def mape(y_true: Iterable[Any], y_pred: Iterable[Any], epsilon: float = 1e-8) -> float:
    truth, pred = _aligned_arrays(y_true, y_pred)
    if not truth.size:
        return float("nan")
    denom = np.maximum(np.abs(truth), epsilon)
    return float(np.mean(np.abs((pred - truth) / denom)))


def smape(y_true: Iterable[Any], y_pred: Iterable[Any], epsilon: float = 1e-8) -> float:
    truth, pred = _aligned_arrays(y_true, y_pred)
    if not truth.size:
        return float("nan")
    denom = np.maximum(np.abs(truth) + np.abs(pred), epsilon)
    return float(np.mean(2.0 * np.abs(pred - truth) / denom))


def directional_accuracy(y_true: Iterable[Any], y_pred: Iterable[Any], last_values: Iterable[Any] | float) -> float:
    truth, pred = _aligned_arrays(y_true, y_pred)
    if not truth.size:
        return float("nan")
    if isinstance(last_values, (int, float, np.integer, np.floating)):
        base = np.full_like(truth, float(last_values), dtype=float)
    else:
        base = finite_array(last_values)
        if base.size != truth.size:
            base = np.resize(base, truth.size)
    true_dir = np.sign(truth - base)
    pred_dir = np.sign(pred - base)
    return float(np.mean(true_dir == pred_dir))


def evaluate_forecast(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    last_values: Iterable[Any] | float | None = None,
) -> dict[str, float]:
    truth, pred = _aligned_arrays(y_true, y_pred)
    out = {
        "mae": mae(truth, pred),
        "rmse": rmse(truth, pred),
        "mape": mape(truth, pred),
        "smape": smape(truth, pred),
    }
    if last_values is not None:
        out["directional_accuracy"] = directional_accuracy(truth, pred, last_values)
    return out


def _aligned_arrays(y_true: Iterable[Any], y_pred: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    truth = finite_array(y_true)
    pred = finite_array(y_pred)
    n = min(truth.size, pred.size)
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    truth = truth[:n]
    pred = pred[:n]
    mask = np.isfinite(truth) & np.isfinite(pred)
    return truth[mask], pred[mask]


def price_log_returns(last_close: float, predictions: Iterable[Any]) -> np.ndarray:
    pred = finite_array(predictions, fallback=max(float(last_close), 1e-6))
    path = np.concatenate([[max(float(last_close), 1e-6)], np.maximum(pred, 1e-6)])
    return np.diff(np.log(path))


def historical_return_diagnostics(target_history: Iterable[Any]) -> dict[str, float]:
    hist = np.maximum(finite_array(target_history), 1e-6)
    if hist.size < 3:
        return {
            "abs_return_q95": 0.0,
            "abs_return_q99": 0.0,
            "abs_return_q995": 0.0,
            "max_abs_return": 0.0,
            "daily_volatility": 0.0,
        }
    returns = np.diff(np.log(hist))
    abs_ret = np.abs(returns[np.isfinite(returns)])
    if not abs_ret.size:
        abs_ret = np.asarray([0.0], dtype=float)
    return {
        "abs_return_q95": float(np.quantile(abs_ret, 0.95)),
        "abs_return_q99": float(np.quantile(abs_ret, 0.99)),
        "abs_return_q995": float(np.quantile(abs_ret, 0.995)),
        "max_abs_return": float(np.max(abs_ret)),
        "daily_volatility": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
    }


def clip_price_path_by_return_bounds(
    predictions: Iterable[Any],
    last_close: float,
    target_history: Iterable[Any],
    multiplier: float = 1.25,
) -> tuple[list[float], dict[str, float]]:
    pred = np.maximum(finite_array(predictions, fallback=max(float(last_close), 1e-6)), 1e-6)
    diagnostics = historical_return_diagnostics(target_history)
    q_bound = diagnostics["abs_return_q995"] * multiplier
    vol_bound = diagnostics["daily_volatility"] * 4.0
    bound = max(q_bound, vol_bound, 1e-4)
    bound = min(bound, max(diagnostics["max_abs_return"] * 1.5, bound), 0.25)

    current = max(float(last_close), 1e-6)
    clipped: list[float] = []
    for value in pred:
        raw_return = math.log(max(float(value), 1e-6) / current)
        clipped_return = float(np.clip(raw_return, -bound, bound))
        current = current * math.exp(clipped_return)
        clipped.append(float(max(current, 1e-6)))
    diagnostics["applied_step_log_return_bound"] = float(bound)
    return clipped, diagnostics


POSITIVE_TERMS = {
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "strong",
    "growth",
    "bullish",
    "outperform",
    "record",
    "surge",
    "rises",
    "gain",
    "gains",
    "positive",
    "buy",
}
NEGATIVE_TERMS = {
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "weak",
    "lawsuit",
    "probe",
    "bearish",
    "underperform",
    "warning",
    "fall",
    "falls",
    "drop",
    "drops",
    "cut",
    "cuts",
    "negative",
    "sell",
}


def lexical_news_signal(headlines: Iterable[str]) -> dict[str, Any]:
    pos = 0
    neg = 0
    hits: list[str] = []
    for headline in headlines:
        text = str(headline or "").lower()
        tokens = set(re.findall(r"[a-z]+", text))
        p = len(tokens & POSITIVE_TERMS)
        n = len(tokens & NEGATIVE_TERMS)
        pos += p
        neg += n
        if p or n:
            hits.append(str(headline)[:240])
    total = pos + neg
    score = 0.0 if total == 0 else (pos - neg) / total
    label = "neutral"
    if score >= 0.35:
        label = "positive"
    elif score <= -0.35:
        label = "negative"
    return {
        "positive_hits": int(pos),
        "negative_hits": int(neg),
        "score": float(score),
        "label": label,
        "evidence_headlines": hits[:8],
    }


def extract_dates_from_text(text: str) -> list[str]:
    matches = re.findall(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?\b", str(text or ""))
    return list(dict.fromkeys(matches))


def model_prediction_summary(
    baseline_predictions: dict[str, dict],
    weights: dict[str, float],
    horizon: int,
    last_close: float = 1.0,
) -> dict[str, Any]:
    """Create human-readable consensus summary of baseline model predictions."""
    names, preds = [], []
    for name in sorted(baseline_predictions.keys()):
        arr = finite_array(baseline_predictions[name].get("predictions", []))
        if arr.size == horizon:
            names.append(name)
            preds.append(arr)
    if not preds:
        return {"consensus_direction": "unknown", "model_count": 0}

    mat = np.vstack(preds)
    mean_pred = np.mean(mat, axis=0)
    std_pred = np.std(mat, axis=0)
    first_mean = float(mean_pred[0])
    last_mean = float(mean_pred[-1])
    consensus = "up" if last_mean > first_mean * 1.002 else ("down" if last_mean < first_mean * 0.998 else "flat")
    max_std_idx = int(np.argmax(std_pred))
    max_disagreement_step = int(max_std_idx)
    max_disagreement_val = float(std_pred[max_std_idx])

    top_indices = np.argsort([-weights.get(n, 0.0) for n in names])[:3]
    top3 = [(names[int(i)], float(weights.get(names[int(i)], 0.0))) for i in top_indices]

    trajectory_summaries = []
    for name in names:
        arr = finite_array(baseline_predictions[name].get("predictions", []))
        if arr.size == horizon:
            start = float(arr[0])
            end = float(arr[-1])
            direction = "up" if end > start * 1.002 else ("down" if end < start * 0.998 else "flat")
            trajectory_summaries.append(f"{name}: {direction} ({start:.2f}->{end:.2f})")

    disagreements = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = float(np.mean(np.abs(preds[i] - preds[j])))
            threshold = 0.03 * max(float(last_close), 1.0)
            if diff > threshold:
                disagreements.append(f"{names[i]} vs {names[j]}: avg ${diff:.2f} apart")

    final_price_range = {
        "min": float(np.min(mat[:, -1])),
        "max": float(np.max(mat[:, -1])),
        "spread_pct": float((np.max(mat[:, -1]) - np.min(mat[:, -1])) / max(last_close, 1e-6) * 100),
    }

    return {
        "model_count": len(names),
        "consensus_direction": consensus,
        "mean_horizon_path": [float(v) for v in mean_pred.tolist()],
        "cross_model_std": [float(v) for v in std_pred.tolist()],
        "max_disagreement_step": max_disagreement_step,
        "max_disagreement_val": max_disagreement_val,
        "top3_by_weight": top3,
        "trajectory_summaries": trajectory_summaries[:8],
        "major_disagreements": disagreements[:4],
        "final_price_range": final_price_range,
    }


def validate_price_prediction(
    predictions: list[float] | np.ndarray,
    last_close: float,
    target_history: list[float],
    forecast_horizon: int,
) -> dict[str, Any]:
    """Validate a price-level prediction for financial sanity."""
    issues: list[str] = []
    pred = np.asarray(predictions, dtype=float)

    if pred.size != forecast_horizon:
        issues.append(f"Length mismatch: {pred.size} vs expected {forecast_horizon}")
    if not np.all(np.isfinite(pred)):
        issues.append("Non-finite values in prediction")
    if pred.size and np.any(pred <= 0):
        issues.append("Non-positive prices")

    scale = float(np.mean(pred)) / max(float(last_close), 1e-6) if pred.size else 0.0
    if pred.size and (scale < 0.2 or scale > 5.0):
        issues.append(f"Price scale suspicious: {scale:.2f}x last close ({last_close:.2f})")

    if pred.size and float(last_close) > 5.0 and float(np.max(np.abs(pred))) < 2.0:
        issues.append("Predictions look like returns rather than price levels")

    returns = price_log_returns(float(last_close), pred) if pred.size else np.array([])
    hist_diag = historical_return_diagnostics(target_history)
    max_step = float(np.max(np.abs(returns))) if returns.size else 0.0
    bound = max(
        hist_diag["abs_return_q995"] * 1.25,
        hist_diag["daily_volatility"] * 4.0,
        1e-4,
    )
    bound = min(bound, max(hist_diag["max_abs_return"] * 1.5, bound), 0.25)
    if max_step > bound + 1e-8:
        issues.append(f"Return step {max_step:.4f} exceeds bound {bound:.4f}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "diagnostics": {
            "scale_ratio": float(scale),
            "max_step_log_return": float(max_step),
            "return_bound": float(bound),
        },
    }


__all__ = [
    "clip_price_path_by_return_bounds",
    "directional_accuracy",
    "evaluate_forecast",
    "extract_dates_from_text",
    "finite_array",
    "historical_return_diagnostics",
    "json_default",
    "lexical_news_signal",
    "mae",
    "mape",
    "model_prediction_summary",
    "parse_jsonish",
    "price_log_returns",
    "rmse",
    "safe_float",
    "smape",
    "validate_price_prediction",
]
