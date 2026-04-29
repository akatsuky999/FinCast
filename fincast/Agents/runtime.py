from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fincast.Agents.baseline_agent import build_baseline_packet
from fincast.Agents.generator_agent import run_generator
from fincast.Agents.investigator_agent import run_investigator
from fincast.Agents.reflector_agent import reflect_forecast
from fincast.tools.dataloader import DEFAULT_MANIFEST_PATH
from fincast.tools.utils import json_default


def run_fincast_pipeline(
    dataset_name: str,
    window_offset: int,
    forecast_horizon: int | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    use_llm_investigator: bool = False,
    use_llm_baseline: bool = False,
    use_llm_generator: bool = False,
    precomputed_baseline_predictions: dict[str, dict] | None = None,
    case_records: list[dict[str, Any]] | None = None,
    training_case_offsets: set[int] | list[int] | tuple[int, ...] | None = None,
    max_generator_retries: int = 2,
) -> dict[str, Any]:
    """Run the full FinCast forecasting pipeline for a single window.

    Stages:
      1. Investigator — data loading + optional LLM news summarization
      2. Baseline — statistical model predictions + case retrieval
      3. Generator — LLM decision-maker (or deterministic fallback)
      4. Reflector — financial sanity validation

    If the Reflector rejects the Generator's prediction, the issues are
    fed back to the LLM for retry (up to max_generator_retries times).
    After all retries are exhausted, falls back to the reference prediction.
    """
    # Stage 1: Investigator
    investigator_packet = run_investigator(
        dataset_name=dataset_name,
        window_offset=window_offset,
        forecast_horizon=forecast_horizon,
        manifest_path=manifest_path,
        use_llm=use_llm_investigator,
    )

    # Stage 2: Baseline
    baseline_packet = build_baseline_packet(
        investigator_packet,
        manifest_path=manifest_path,
        use_llm=use_llm_baseline,
        precomputed_baseline_predictions=precomputed_baseline_predictions,
        case_records=case_records,
        training_case_offsets=training_case_offsets,
    )

    # Stage 3: Generator (with optional LLM retry loop)
    generator_packet = run_generator(
        investigator_packet=investigator_packet,
        baseline_packet=baseline_packet,
        manifest_path=manifest_path,
        use_llm=use_llm_generator,
        max_retries=max_generator_retries,
    )

    # Stage 4: Reflector
    reflector_report = reflect_forecast(
        generator_packet, investigator_packet, baseline_packet
    )

    # If Reflector rejected and we have retries left, feed issues back to Generator
    retries_used = 0
    while not reflector_report.get("approved", False) and retries_used < max_generator_retries:
        retries_used += 1
        issues = reflector_report.get("issues", [])
        # Append rejection feedback to investigator packet for LLM context
        investigator_packet.setdefault("_reflector_feedback", [])
        investigator_packet["_reflector_feedback"].append({
            "attempt": retries_used,
            "issues": issues,
            "warnings": reflector_report.get("warnings", []),
        })

        generator_packet = run_generator(
            investigator_packet=investigator_packet,
            baseline_packet=baseline_packet,
            manifest_path=manifest_path,
            use_llm=use_llm_generator,
            max_retries=0,  # Single attempt within each retry
        )
        reflector_report = reflect_forecast(
            generator_packet, investigator_packet, baseline_packet
        )

    # If still rejected after all retries, fall back to reference prediction
    if not reflector_report.get("approved", False):
        generator_packet["final_prediction"] = list(baseline_packet["reference_prediction"])
        generator_packet["confidence"] = min(
            float(generator_packet.get("confidence", 0.0) or 0.0), 0.40
        )
        generator_packet.setdefault("warnings", []).append(
            f"Reflector rejected after {retries_used} retries; "
            f"falling back to reference prediction. "
            f"Issues: {'; '.join(reflector_report.get('issues', []))}"
        )
        reflector_report = reflect_forecast(
            generator_packet, investigator_packet, baseline_packet
        )

    return {
        "dataset": dataset_name,
        "ticker": investigator_packet.get("ticker"),
        "window_offset": int(window_offset),
        "forecast_horizon": int(investigator_packet["forecast_horizon"]),
        "prediction_timestamps": investigator_packet["prediction_timestamps"],
        "final_prediction": generator_packet["final_prediction"],
        "confidence": generator_packet.get("confidence"),
        "approved": bool(reflector_report.get("approved", False)),
        "investigator_packet": investigator_packet,
        "baseline_packet": baseline_packet,
        "generator_packet": generator_packet,
        "reflector_report": reflector_report,
        "retries_used": retries_used,
    }


def main() -> None:
    dataset = os.getenv("FINCAST_DATASET", "FinCastPrice_NVDA")
    offset = int(os.getenv("FINCAST_WINDOW_OFFSET", "1500"))
    horizon = int(os.getenv("FINCAST_FORECAST_HORIZON", "5"))
    use_llm = os.getenv("FINCAST_USE_LLM", "0") == "1"
    packet = run_fincast_pipeline(
        dataset,
        offset,
        horizon,
        use_llm_investigator=use_llm,
        use_llm_baseline=False,
        use_llm_generator=use_llm,
    )
    compact = {
        "dataset": packet["dataset"],
        "window_offset": packet["window_offset"],
        "approved": packet["approved"],
        "final_prediction": packet["final_prediction"],
        "confidence": packet["confidence"],
        "reflector_report": packet["reflector_report"],
        "retries_used": packet["retries_used"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()


__all__ = ["run_fincast_pipeline"]
