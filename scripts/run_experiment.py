from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from fincast.Agents.runtime import run_fincast_pipeline
from fincast.tools.dataloader import FinCastDataLoader
from fincast.tools.utils import json_default, safe_float


def load_experiment_config(config_path: str | Path) -> dict:
    """Load experiment configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_datasets(cfg: dict) -> list[str]:
    """Resolve dataset list from config, falling back to manifest if empty."""
    datasets = cfg.get("datasets") or []
    if not datasets:
        manifest_path = PROJECT_ROOT / cfg.get("manifest_path", "FinCast/Data/Datasets_return/manifest_fincast_price.yaml")
        loader = FinCastDataLoader(manifest_path)
        datasets = [entry.name for entry in loader.datasets]
    return datasets


def print_single_window_result(packet: dict) -> None:
    """Print detailed results for a single prediction window."""
    diag = packet["baseline_packet"]["baseline_diagnostics"]
    gen = packet["generator_packet"]
    refl = packet["reflector_report"]

    print("\n" + "─" * 50)
    print("Prediction Summary")
    print("─" * 50)
    print(f"  Dataset:      {packet['dataset']}")
    print(f"  Window offset:{packet['window_offset']}")
    print(f"  Horizon:      {packet['forecast_horizon']}")
    print(f"  Reflector:    {'PASS' if packet['approved'] else 'REJECT'}")
    print(f"  Confidence:   {packet['confidence']:.4f}")
    print(f"  Retries:      {packet['retries_used']}")

    print(f"\n  Cluster match similarity: {diag['cluster_match_similarity']:.4f}")
    print(f"  Adaptive weights:         w_cluster={diag['cluster_weight']:.2f}  w_case={diag['case_weight']:.2f}")
    print(f"  Models run:               {len(diag['models_run'])}/12")
    print(f"  Cluster voted:            {', '.join(diag['cluster_voted_models'])}")
    print(f"  Weighting method:         {diag['weighting_method']}")

    ref = packet["baseline_packet"]["reference_prediction"]
    pred = packet["final_prediction"]
    print(f"\n  Reference [0:5]:  {[round(v, 3) for v in ref[:5]]}")
    print(f"  Reference [-5:]:  {[round(v, 3) for v in ref[-5:]]}")
    print(f"  Final pred [0:5]: {[round(v, 3) for v in pred[:5]]}")
    print(f"  Final pred [-5:]: {[round(v, 3) for v in pred[-5:]]}")

    if gen.get("adjustment_reason"):
        ar = gen["adjustment_reason"]
        print(f"\n  Generator adjustment:")
        print(f"    Policy:      {ar.get('policy', 'N/A')}")
        print(f"    Direction:   {ar.get('selected_direction', 'N/A')}")
        print(f"    Strength:    {safe_float(ar.get('selected_strength'), 0):.4f}")
        print(f"    Evidence:    {ar.get('evidence', [])}")

    if refl.get("issues"):
        print(f"\n  Reflector issues: {refl['issues']}")
    if refl.get("warnings"):
        print(f"  Reflector warnings: {refl['warnings']}")

    print("─" * 50)


def run_single_window(cfg: dict, dataset: str, offset: int) -> None:
    """Single-window debug mode."""
    manifest_path = str(PROJECT_ROOT / cfg.get("manifest_path", "FinCast/Data/Datasets_return/manifest_fincast_price.yaml"))
    horizon = cfg.get("predicted_window", 60)

    print(f"Single-window mode: {dataset} offset={offset} horizon={horizon}")
    print(f"LLM Investigator: {cfg.get('use_llm_investigator', False)}")
    print(f"LLM Generator:    {cfg.get('use_llm_generator', False)}")

    packet = run_fincast_pipeline(
        dataset_name=dataset,
        window_offset=offset,
        forecast_horizon=horizon,
        manifest_path=manifest_path,
        use_llm_investigator=cfg.get("use_llm_investigator", False),
        use_llm_baseline=False,
        use_llm_generator=cfg.get("use_llm_generator", False),
        max_generator_retries=cfg.get("max_generator_retries", 2),
    )
    print_single_window_result(packet)


def _run_one_dataset(
    dataset: str,
    manifest_path: str,
    output_dir: str,
    test_ratio: float,
    seed: int,
    use_llm_generator: bool,
    force_rebuild_cases: bool,
    max_test_windows: int | None,
    concurrency: int,
) -> dict[str, str]:
    """Run benchmark for a single dataset."""
    from fincast.tools.evaluate_pipeline import run_random_sample_benchmark

    return run_random_sample_benchmark(
        manifest_path=manifest_path, output_dir=output_dir,
        test_ratio=test_ratio, seed=seed, datasets=[dataset],
        use_llm_generator=use_llm_generator,
        force_rebuild_cases=force_rebuild_cases,
        max_test_windows=max_test_windows,
        concurrency=concurrency,
    )


def run_benchmark(cfg: dict) -> None:
    """Full benchmark mode (supports parallel datasets + concurrent windows)."""
    manifest_path = str(PROJECT_ROOT / cfg.get("manifest_path", "FinCast/Data/Datasets_return/manifest_fincast_price.yaml"))
    output_dir = str(PROJECT_ROOT / cfg.get("output_dir", "FinCast/outputs"))
    datasets = resolve_datasets(cfg) if not cfg.get("datasets") else cfg["datasets"]
    parallel = cfg.get("parallel", False)
    test_ratio = cfg.get("test_ratio", 0.20)
    split_seed = cfg.get("split_seed", 5053)
    use_llm = cfg.get("use_llm_generator", False)
    force = cfg.get("force_rebuild_cases", False)
    max_w = cfg.get("max_test_windows")
    concurrency = cfg.get("test_concurrency", 1)

    # Sanity check for LLM concurrency
    if use_llm and concurrency > 1:
        try:
            from fincast.Agents.generator_agent import api_key_count
            n_keys = api_key_count()
        except Exception:
            n_keys = 1
        per_key = concurrency / n_keys
        print(f"  API keys: {n_keys} | concurrency: {concurrency} ({per_key:.0f}/key)", file=sys.stderr)
        if per_key > 8:
            print(f"  Warning: high load per key. Recommended concurrency <= {n_keys * 6}", file=sys.stderr)

    print("Full Benchmark Mode")
    print(f"  Datasets:     {', '.join(datasets)}")
    print(f"  Test ratio:   {test_ratio}")
    print(f"  LLM Generator: {use_llm}")
    print(f"  Parallel:     {parallel}")
    print(f"  Output:       {output_dir}")

    if parallel and len(datasets) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"\n  Launching {len(datasets)} parallel tasks...")
        dataset_seeds = {ds: split_seed + i * 1009 for i, ds in enumerate(datasets)}
        all_paths: dict[str, dict[str, str]] = {}

        with ThreadPoolExecutor(max_workers=len(datasets)) as executor:
            futures = {
                executor.submit(
                    _run_one_dataset, ds, manifest_path, output_dir,
                    test_ratio, dataset_seeds[ds], use_llm, force, max_w, concurrency,
                ): ds
                for ds in datasets
            }
            for future in as_completed(futures):
                ds = futures[future]
                try:
                    paths = future.result()
                    all_paths[ds] = paths
                    print(f"  OK  {ds}")
                except Exception as exc:
                    print(f"  FAIL {ds}: {exc}")

        print(f"\n  Benchmark complete. Output files:")
        for ds, paths in all_paths.items():
            for name, path in paths.items():
                print(f"    {ds}/{name}: {path}")
    else:
        from fincast.tools.evaluate_pipeline import run_random_sample_benchmark

        paths = run_random_sample_benchmark(
            manifest_path=manifest_path, output_dir=output_dir,
            test_ratio=test_ratio, seed=split_seed,
            datasets=datasets if datasets else None,
            use_llm_generator=use_llm, force_rebuild_cases=force,
            max_test_windows=max_w, concurrency=concurrency,
        )
        print(f"\n  Benchmark complete. Output files:")
        for name, path in paths.items():
            print(f"    {name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinCast Experiment — Single-Window Debug / Full Benchmark",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PACKAGE_ROOT / "scripts" / "experiment_config.yaml"),
        help="Path to experiment config file (YAML)",
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="Single-window mode: specify window_offset (overrides config)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to evaluate (overrides config)",
    )
    parser.add_argument(
        "--use-llm-generator", action="store_true", default=None,
        help="Enable LLM Generator Agent",
    )
    parser.add_argument(
        "--parallel", action="store_true", default=None,
        help="Run all datasets in parallel (overrides config)",
    )
    parser.add_argument(
        "--test-concurrency", type=int, default=None,
        help="Concurrent windows per dataset (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)

    if args.datasets is not None:
        cfg["datasets"] = args.datasets
    if args.use_llm_generator is not None:
        cfg["use_llm_generator"] = args.use_llm_generator
    if args.parallel is not None:
        cfg["parallel"] = args.parallel
    if args.test_concurrency is not None:
        cfg["test_concurrency"] = args.test_concurrency
    if args.window is not None:
        cfg["single_window_offset"] = args.window

    single_offset = cfg.get("single_window_offset")
    if single_offset is not None:
        datasets = resolve_datasets(cfg)
        run_single_window(cfg, datasets[0], int(single_offset))
    else:
        run_benchmark(cfg)


if __name__ == "__main__":
    main()
