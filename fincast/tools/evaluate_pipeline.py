from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fincast.Agents.runtime import run_fincast_pipeline
from fincast.tools.caselib import analyze_training, load_case_library
from fincast.tools.dataloader import DEFAULT_MANIFEST_PATH, FinCastDataLoader
from fincast.tools.utils import evaluate_forecast, json_default, safe_float


BASELINE_ORDER = [
    "LastClose",
    "RandomWalkDrift",
    "EWMAReturn",
    "ARReturn",
    "ARMAReturn",
    "ARIMALogPrice",
    "ARIMAXPrice",
    "ARMAGARCHReturn",
    "SeasonalNaive",
    "HistoricAverage",
    "AutoETS",
    "Theta",
]
FINCAST_MODEL_NAME = "FinCast"


def _sample_offsets(cases: list[dict[str, Any]], test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    offsets = np.asarray([int(case["window_offset"]) for case in cases], dtype=int)
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(offsets) * float(test_ratio))))
    test_offsets = set(int(v) for v in rng.choice(offsets, size=n_test, replace=False))
    train_offsets = [int(v) for v in offsets if int(v) not in test_offsets]
    return sorted(train_offsets), sorted(test_offsets)


def _last_close_from_case(case: dict[str, Any]) -> float:
    try:
        return float(case["baseline_predictions"]["LastClose"]["predictions"][0])
    except Exception:
        truth = case.get("truth") or []
        return safe_float(truth[0] if truth else 0.0, 0.0)


def _process_one_window(
    dataset: str, ticker: str, offset: int, horizon: int,
    manifest_path: str | Path, case: dict[str, Any],
    cases: list[dict[str, Any]], train_offset_set: set[int],
    use_llm_strategist: bool, verbose: bool = False,
) -> list[dict[str, Any]]:
    """Process a single test window. Returns list of prediction rows."""
    import time as _time
    t0 = _time.time()
    packet = run_fincast_pipeline(
        dataset, int(offset), int(horizon),
        manifest_path=manifest_path,
        use_llm_briefing=False, use_llm_baseline=False,
        use_llm_strategist=use_llm_strategist,
        precomputed_baseline_predictions=case["baseline_predictions"],
        case_records=cases,
        training_case_offsets=train_offset_set,
        max_strategist_retries=2,
    )
    elapsed = _time.time() - t0

    rows: list[dict[str, Any]] = []
    _append_forecast_rows(
        rows, dataset, ticker, FINCAST_MODEL_NAME,
        int(offset), packet["prediction_timestamps"],
        [float(v) for v in case["truth"]], packet["final_prediction"],
        _last_close_from_case(case),
        approved=packet["approved"],
        confidence=safe_float(packet.get("confidence"), 0.0),
    )
    for model in BASELINE_ORDER:
        forecast = case["baseline_predictions"].get(model)
        if not forecast:
            continue
        _append_forecast_rows(
            rows, dataset, ticker, model,
            int(offset), packet["prediction_timestamps"],
            [float(v) for v in case["truth"]],
            [float(v) for v in forecast["predictions"]],
            _last_close_from_case(case),
        )

    if verbose:
        status = "OK" if packet["approved"] else "FAIL"
        print(f"  [{ticker:>4s} #{offset:>4d}] {elapsed:>5.0f}s {status}",
              file=sys.stderr, flush=True)
    return rows


def _append_forecast_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    ticker: str,
    model: str,
    window_offset: int,
    prediction_timestamps: list[str],
    truth: list[float],
    predictions: list[float],
    last_close: float,
    approved: bool | None = None,
    confidence: float | None = None,
) -> None:
    for h, (ts, actual, pred) in enumerate(zip(prediction_timestamps, truth, predictions), start=1):
        row = {
            "dataset": dataset,
            "ticker": ticker,
            "model": model,
            "window_offset": int(window_offset),
            "horizon_index": h,
            "time_stamp": ts,
            "actual": float(actual),
            "prediction": float(pred),
            "last_close": float(last_close),
        }
        if approved is not None:
            row["approved"] = bool(approved)
        if confidence is not None:
            row["confidence"] = float(confidence)
        rows.append(row)


def _metrics_from_rows(prediction_rows: list[dict[str, Any]], split_meta: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "model"]
    for (dataset, model), group in df.groupby(group_cols, sort=True):
        metrics = evaluate_forecast(group["actual"], group["prediction"], group["last_close"])
        metric_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n_windows": int(group["window_offset"].nunique()),
                "n_points": int(len(group)),
                **metrics,
                **split_meta.get(str(dataset), {}),
            }
        )

    for model, group in df.groupby("model", sort=True):
        metrics = evaluate_forecast(group["actual"], group["prediction"], group["last_close"])
        metric_rows.append(
            {
                "dataset": "ALL",
                "model": model,
                "n_windows": int(group[["dataset", "window_offset"]].drop_duplicates().shape[0]),
                "n_points": int(len(group)),
                **metrics,
                "test_ratio": split_meta.get("_global", {}).get("test_ratio"),
                "split_seed": split_meta.get("_global", {}).get("split_seed"),
                "split_unit": "sliding_window_sample",
            }
        )
    out = pd.DataFrame(metric_rows)
    order = {name: idx for idx, name in enumerate([FINCAST_MODEL_NAME, *BASELINE_ORDER])}
    out["_model_order"] = out["model"].map(lambda name: order.get(str(name), 999))
    out["_dataset_order"] = out["dataset"].map(lambda name: 999 if name == "ALL" else 0)
    out = out.sort_values(["_dataset_order", "dataset", "_model_order", "model"]).drop(columns=["_model_order", "_dataset_order"])
    return out


def run_random_sample_benchmark(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = "FinCast/outputs",
    test_ratio: float = 0.20,
    seed: int = 5053,
    datasets: list[str] | None = None,
    use_llm_strategist: bool = False,
    force_rebuild_cases: bool = False,
    max_test_windows: int | None = None,
    concurrency: int = 1,
) -> dict[str, str]:
    loader = FinCastDataLoader(manifest_path)
    dataset_names = datasets or [entry.name for entry in loader.datasets]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_meta: dict[str, Any] = {"_global": {"test_ratio": float(test_ratio), "split_seed": int(seed)}}

    for ds_index, dataset in enumerate(dataset_names):
        entry = loader.resolve_dataset(dataset)
        ticker = entry.name.split("_")[-1]
        print(f"\n[FinCast] Training phase: analyze_training for {dataset}...", file=sys.stderr)
        analyze_training(dataset, manifest_path, force=force_rebuild_cases)
        metadata, cases, _ = load_case_library(dataset, manifest_path)
        train_offsets, test_offsets = _sample_offsets(cases, test_ratio, seed + ds_index * 1009)
        if max_test_windows is not None:
            test_offsets = test_offsets[: int(max_test_windows)]
        train_offset_set = set(train_offsets)
        case_by_offset = {int(case["window_offset"]): case for case in cases}
        split_meta[dataset] = {
            "test_ratio": float(test_ratio),
            "split_seed": int(seed + ds_index * 1009),
            "split_unit": "sliding_window_sample",
            "look_back": int(metadata.get("look_back", loader.look_back)),
            "horizon": int(metadata.get("horizon", loader.predicted_window)),
            "stride": int(metadata.get("stride", loader.manifest.get("sliding_window", loader.predicted_window))),
            "train_windows": int(len(train_offsets)),
            "test_windows": int(len(test_offsets)),
        }
        for offset in train_offsets:
            split_rows.append({"dataset": dataset, "ticker": ticker, "window_offset": offset, "split": "train"})
        for offset in test_offsets:
            split_rows.append({"dataset": dataset, "ticker": ticker, "window_offset": offset, "split": "test"})

        # Test window evaluation
        n_test = len(test_offsets)
        horizon = int(metadata.get("horizon", loader.predicted_window))
        width = 40
        print(f"  Evaluating {dataset}: {len(train_offsets)} train / {n_test} test windows"
              + (f" (concurrency={concurrency})" if concurrency > 1 else ""), file=sys.stderr)

        if concurrency <= 1:
            # ── Sequential mode (original) ──
            for idx, offset in enumerate(test_offsets, start=1):
                rows = _process_one_window(
                    dataset, ticker, offset, horizon, manifest_path,
                    case_by_offset[int(offset)], cases, train_offset_set,
                    use_llm_strategist,
                )
                prediction_rows.extend(rows)
                if idx % max(1, n_test // 100) == 0 or idx == n_test:
                    pct = idx / n_test
                    filled = int(width * pct)
                    bar = "█" * filled + "░" * (width - filled)
                    print(f"\r  [{bar}] {idx}/{n_test}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
        else:
            # ── Concurrent mode ──
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Detect API key pool size (module-level, available before any LLM call)
            try:
                from fincast.Agents.strategist_agent import api_key_count
                n_keys = api_key_count()
            except Exception:
                n_keys = 1

            print(f"  workers={concurrency}  keys={n_keys}  windows={n_test}",
                  file=sys.stderr)
            if n_keys > 1:
                print(f"  Key pool: {n_keys} keys, requests distributed round-robin",
                      file=sys.stderr)

            completed = 0
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _process_one_window, dataset, ticker, offset, horizon,
                        manifest_path, case_by_offset[int(offset)], cases,
                        train_offset_set, use_llm_strategist, True,  # verbose=True
                    ): offset
                    for offset in test_offsets
                }
                for future in as_completed(futures):
                    try:
                        rows = future.result()
                        prediction_rows.extend(rows)
                    except Exception as exc:
                        print(f"\n  [warn] window {futures[future]} failed: {exc}", file=sys.stderr)
                    completed += 1
                    if completed % max(1, n_test // 20) == 0 or completed == n_test:
                        pct = completed / n_test
                        filled = int(width * pct)
                        bar = "█" * filled + "░" * (width - filled)
                        # Count unique active threads (approximate)
                        alive = sum(1 for f in futures if not f.done())
                        print(f"\r  [{bar}] {completed}/{n_test} (≈{alive} active)", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)

    predictions_df = pd.DataFrame(prediction_rows)
    split_df = pd.DataFrame(split_rows)
    metrics_df = _metrics_from_rows(prediction_rows, split_meta)

    # ── Per-dataset output ──
    output_paths: dict[str, str] = {}
    dataset_names_in_run = sorted(set(
        str(r["dataset"]) for r in prediction_rows
        if str(r["dataset"]) != "ALL"
    ))
    for ds_name in dataset_names_in_run:
        ds_dir = out_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        ds_pred = predictions_df[predictions_df["dataset"] == ds_name]
        ds_split = split_df[split_df["dataset"] == ds_name] if "dataset" in split_df.columns else pd.DataFrame()
        ds_metrics = metrics_df[metrics_df["dataset"] == ds_name]

        pred_path = ds_dir / "predictions.csv"
        split_path_ds = ds_dir / "split.csv"
        metrics_path_ds = ds_dir / "metrics.csv"

        ds_pred.to_csv(pred_path, index=False)
        if not ds_split.empty:
            ds_split.to_csv(split_path_ds, index=False)
        ds_metrics.to_csv(metrics_path_ds, index=False)

        print(f"  Saved {ds_name} → {ds_dir}", file=sys.stderr)

    # ── Aggregate output (all datasets) ──
    all_dir = out_dir / "_summary"
    all_dir.mkdir(parents=True, exist_ok=True)
    metrics_all = all_dir / "metrics.csv"
    predictions_all = all_dir / "predictions.csv"
    split_all = all_dir / "split.csv"
    meta_all = all_dir / "meta.json"
    metrics_df.to_csv(metrics_all, index=False)
    predictions_df.to_csv(predictions_all, index=False)
    split_df.to_csv(split_all, index=False)
    meta_all.write_text(json.dumps(split_meta, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    output_paths["metrics_csv"] = str(metrics_all)
    output_paths["predictions_csv"] = str(predictions_all)
    output_paths["split_csv"] = str(split_all)
    output_paths["meta_json"] = str(meta_all)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinCast random-sample benchmark on Datasets_return.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default="FinCast/outputs")
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=5053)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--use-llm-strategist", action="store_true")
    parser.add_argument("--force-rebuild-cases", action="store_true")
    parser.add_argument("--max-test-windows", type=int, default=None)
    args = parser.parse_args()
    paths = run_random_sample_benchmark(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        datasets=args.datasets,
        use_llm_strategist=args.use_llm_strategist,
        force_rebuild_cases=args.force_rebuild_cases,
        max_test_windows=args.max_test_windows,
    )
    print(json.dumps(paths, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = ["BASELINE_ORDER", "FINCAST_MODEL_NAME", "run_random_sample_benchmark"]
