from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from fincast.tools.caselib import analyze_training
from fincast.tools.dataloader import FinCastDataLoader


def load_train_config(config_path: str | Path) -> dict:
    """Load training configuration file."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinCast Training — Build Case Library & K-Medoids Clusters",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PACKAGE_ROOT / "scripts" / "train_config.yaml"),
        help="Path to training config file (YAML)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to train (overrides config)",
    )
    parser.add_argument(
        "--force", action="store_true", default=None,
        help="Force rebuild case library (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)

    manifest_path = str(PROJECT_ROOT / cfg.get("manifest_path", "FinCast/Data/Datasets_return/manifest_fincast_price.yaml"))
    datasets = args.datasets if args.datasets else resolve_datasets(cfg)
    force = args.force if args.force is not None else cfg.get("force_rebuild", False)

    print("=" * 60)
    print("FinCast Training Phase")
    print("=" * 60)
    print(f"Config:    {args.config}")
    print(f"Manifest:  {manifest_path}")
    print(f"Datasets:  {', '.join(datasets)}")
    print(f"Params:    look_back={cfg.get('look_back')}, "
          f"predicted_window={cfg.get('predicted_window')}, "
          f"sliding_window={cfg.get('sliding_window')}")
    print(f"Clustering: num_clusters={cfg.get('num_clusters')}, "
          f"method={cfg.get('cluster_method')}, "
          f"min_votes={cfg.get('min_votes')}")
    print(f"Force rebuild: {force}")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n>>> Training {dataset} ...")
        try:
            cache_path = analyze_training(
                dataset_name=dataset,
                manifest_path=manifest_path,
                force=force,
            )
            print(f"    Done → {cache_path}")
        except Exception as exc:
            print(f"    Failed: {exc}")
            if len(datasets) == 1:
                raise

    print("\n" + "=" * 60)
    print("Training complete. Run: python scripts/run_experiment.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
