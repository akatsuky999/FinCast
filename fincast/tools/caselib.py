from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fincast.basemodels import BASELINE_VERSION, run_all_baselines
from fincast.tools.dataloader import DEFAULT_MANIFEST_PATH, FinCastDataLoader
from fincast.tools.similarity import zscore


CASELIB_VERSION = "fincast_case_library_v1"
CASE_STORAGE_DIR = Path(__file__).resolve().parents[1] / "caselib"
CACHE_DIR = CASE_STORAGE_DIR / "cache"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def case_cache_path(dataset_name: str, look_back: int, horizon: int, stride: int) -> Path:
    safe = dataset_name.replace("/", "_")
    return CACHE_DIR / f"{safe}_lb{look_back}_h{horizon}_s{stride}.jsonl"


def _metrics(pred: list[float], truth: np.ndarray, last_close: float) -> dict[str, float]:
    arr = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if arr.size != truth.size or not np.isfinite(arr).all() or not np.isfinite(truth).all():
        return {"mse": 1e12, "mae": 1e6, "directional_accuracy": 0.0}
    err = np.clip(arr - truth, -1e6, 1e6)
    final_pred_dir = np.sign(arr[-1] - last_close)
    final_true_dir = np.sign(truth[-1] - last_close)
    return {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "directional_accuracy": float(final_pred_dir == final_true_dir),
    }


def _case_features(packet: dict[str, Any]) -> dict[str, float]:
    feats = packet.get("financial_features", {})
    keys = [
        "cumulative_log_return",
        "realized_volatility_daily",
        "realized_volatility_annualized",
        "max_drawdown",
        "skewness",
        "kurtosis",
        "trend_slope_log_price",
        "volume_spike_ratio_20d",
        "news_total_lookback",
        "news_density_lookback",
        "last_return_lag1",
        "high_low_range_last",
        "open_close_gap_last",
        "mean_daily_log_return",
        "news_active_days_lookback",
    ]
    return {k: float(feats.get(k, 0.0) or 0.0) for k in keys}


def _summarize_future(truth: np.ndarray, last_close: float) -> str:
    if truth.size == 0 or last_close <= 0:
        return "Unknown"
    ret = float(np.log(float(truth[-1]) / last_close))
    direction = "up" if ret > 0.01 else ("down" if ret < -0.01 else "flat")
    vol = float(np.std(np.diff(np.log(np.maximum(truth, 1e-6)))) if truth.size > 1 else 0.0)
    return (
        f"Direction: {direction}, Total_return: {ret*100:.1f}%, "
        f"Realized_vol: {vol*100:.2f}%, "
        f"Price_range: [{float(np.min(truth)):.2f}, {float(np.max(truth)):.2f}]"
    )


def _metadata(
    loader: FinCastDataLoader,
    dataset_name: str,
    look_back: int,
    horizon: int,
    stride: int,
    max_cases: int | None = None,
) -> dict[str, Any]:
    entry = loader.resolve_dataset(dataset_name)
    return {
        "type": "metadata",
        "case_library_version": CASELIB_VERSION,
        "baseline_version": BASELINE_VERSION,
        "dataset_name": entry.name,
        "full_csv": str(entry.full_csv),
        "full_csv_sha256": _file_hash(entry.full_csv),
        "full_csv_mtime": entry.full_csv.stat().st_mtime,
        "look_back": int(look_back),
        "horizon": int(horizon),
        "stride": int(stride),
        "max_cases": max_cases,
    }


def build_case_library(
    dataset_name: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    look_back: int | None = None,
    horizon: int | None = None,
    stride: int | None = None,
    force: bool = False,
    max_cases: int | None = None,
) -> Path:
    loader = FinCastDataLoader(manifest_path)
    entry = loader.resolve_dataset(dataset_name)
    full = pd.read_csv(entry.full_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    L = int(look_back or loader.look_back)
    H = int(horizon or loader.predicted_window)
    S = int(stride or loader.manifest.get("sliding_window", H))
    path = case_cache_path(entry.name, L, H, S)
    path.parent.mkdir(parents=True, exist_ok=True)

    max_start = len(full) - (L + H)
    expected_cases = len(range(0, max_start + 1, S)) if max_start >= 0 else 0
    if max_cases is not None:
        expected_cases = min(expected_cases, int(max_cases))
    meta = _metadata(loader, entry.name, L, H, S, max_cases=max_cases)
    meta["expected_cases"] = expected_cases
    if path.exists() and not force:
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
                cached_rows = sum(1 for line in f if line.strip())
            comparable = {k: first.get(k) for k in meta.keys() if k != "full_csv_mtime"}
            expected = {k: meta.get(k) for k in meta.keys() if k != "full_csv_mtime"}
            legacy_meta = "expected_cases" not in first
            if legacy_meta:
                comparable.pop("expected_cases", None)
                expected.pop("expected_cases", None)
            if comparable == expected and cached_rows == expected_cases:
                return path
        except Exception:
            pass

    # Simple progress bar helper
    def _progress(iterable, total, label="Building cases"):
        from math import ceil
        width = 40
        for i, item in enumerate(iterable):
            pct = (i + 1) / total
            filled = int(width * pct)
            bar = "█" * filled + "░" * (width - filled)
            print(f"\r  {label}: |{bar}| {i+1}/{total} ({pct*100:.0f}%)", end="", file=sys.stderr, flush=True)
            yield item
        print(file=sys.stderr)

    offsets = list(range(0, max_start + 1, S))
    if max_cases is not None:
        offsets = offsets[:max_cases]
    total = len(offsets)
    print(f"  Building case library: {entry.name} (stride={S}, {total} cases)", file=sys.stderr)

    rows_written = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, default=_json_default) + "\n")
        for offset in _progress(offsets, total):
            packet = loader.gather_forecast_inputs(entry.name, offset, H, L)
            window = full.iloc[offset : offset + L].copy()
            truth = full.iloc[offset + L : offset + L + H]["target_close"].to_numpy(dtype=float)
            preds = run_all_baselines(window, H, packet["prediction_timestamps"])
            metrics = {
                name: _metrics(forecast["predictions"], truth, packet["financial_features"]["last_close"])
                for name, forecast in preds.items()
            }
            best_model = min(metrics.items(), key=lambda kv: kv[1]["mse"])[0]
            target_series = window["target_close"].to_numpy(dtype=float)
            case = {
                "type": "case",
                "dataset": entry.name,
                "ticker": packet["ticker"],
                "window_offset": offset,
                "look_back_start": packet["look_back_start"],
                "look_back_end": packet["look_back_end"],
                "future_start": packet["prediction_start"],
                "future_end": packet["prediction_timestamps"][-1],
                "prediction_timestamps": packet["prediction_timestamps"],
                "truth": truth.tolist(),
                "raw_window": target_series.tolist(),
                "zscored_window": zscore(target_series).tolist(),
                "baseline_predictions": preds,
                "metrics": metrics,
                "best_model": best_model,
                "case_features": _case_features(packet),
                "future_performance_summary": _summarize_future(truth, packet["financial_features"]["last_close"]),
                "news_context_snapshot": {
                    "top_headlines": packet.get("news_context", {}).get("top_headlines", [])[:5],
                    "combined_text": packet.get("news_context", {}).get("combined_recent_text", "")[:1000],
                },
            }
            f.write(json.dumps(case, default=_json_default, allow_nan=False) + "\n")
            rows_written += 1

    # Build clusters from all cases and append to JSONL
    all_cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "case" and obj.get("zscored_window"):
                all_cases.append(obj)

    if len(all_cases) >= 6:
        try:
            from fincast.Agents.cluster import cluster_cases  # lazy to avoid circular import

            num_clusters = min(6, max(3, len(all_cases) // 20))
            clusters = cluster_cases(all_cases, num_clusters=num_clusters, method="weighted")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "clusters",
                    "clusters": clusters,
                    "num_clusters": num_clusters,
                    "method": "weighted",
                }, default=_json_default, allow_nan=False) + "\n")
            print(f"  Built {len(clusters)} clusters from {len(all_cases)} cases.", file=sys.stderr)
        except Exception as exc:
            print(f"  [warn] Clustering failed: {exc}", file=sys.stderr)

    return path


def analyze_training(
    dataset_name: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    force: bool = False,
) -> Path:
    """One-time training phase: build case library + K-Medoids clusters.

    Runs all 12 baseline models on every sliding window of the full dataset,
    records MSE scores, determines best model per window, and clusters cases.

    This is the equivalent of AlphaCast's analyze_training(). Call once per
    dataset before running any predictions.
    """
    return build_case_library(dataset_name, manifest_path, force=force)


def load_case_library(
    dataset_name: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    look_back: int | None = None,
    horizon: int | None = None,
    stride: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load pre-built case library and clusters. Fails if not yet built.

    Call analyze_training() first to build the library.
    """
    loader = FinCastDataLoader(manifest_path)
    entry = loader.resolve_dataset(dataset_name)
    L = int(look_back or loader.look_back)
    H = int(horizon or loader.predicted_window)
    S = int(stride or loader.manifest.get("sliding_window", H))
    path = case_cache_path(entry.name, L, H, S)
    if not path.exists():
        raise FileNotFoundError(
            f"Case library not found: {path}\n"
            f"Run analyze_training('{dataset_name}') first."
        )
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    metadata = lines[0] if lines else {}
    cases = [line for line in lines[1:] if line.get("type") == "case"]
    clusters = []
    for line in lines[1:]:
        if line.get("type") == "clusters":
            clusters = line.get("clusters", [])
            break
    return metadata, cases, clusters
