from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd

from fincast.Agents.cluster import match_cluster
from fincast.basemodels import get_baseline_models, run_all_baselines
from fincast.tools.caselib import load_case_library
from fincast.tools.dataloader import DEFAULT_MANIFEST_PATH, FinCastDataLoader
from fincast.tools.similarity import comprehensive_similarity, retrieve_similar_cases, zscore


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PACKAGE_ROOT / ".env"


BASELINE_AGENT_PROMPT = dedent(
    """
    You are BaselineAgent for FinCast.

    You explain a deterministic baseline-selection packet. You must not change
    model weights, predictions, reference_prediction, or timestamps.

    Return a compact JSON object with exactly:
      - model_selection_summary
      - case_evidence_summary
      - caution_notes

    Never output a new forecast. Never invent metrics or future information.
    """
).strip()


def _load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _model_name_from_env() -> str | None:
    _load_env_file()
    model_name = os.getenv("PYA_MODEL")
    if not model_name:
        model_raw = os.getenv("MODEL")
        if model_raw:
            model_name = f"openai:{model_raw}"
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = base_url
    if not model_name or ":" not in model_name:
        return None
    return model_name


def _parse_jsonish(text: str) -> dict[str, Any]:
    raw = text.strip()
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
        return {"model_selection_summary": text.strip(), "case_evidence_summary": "", "caution_notes": ""}
    return {
        "model_selection_summary": str(parsed.get("model_selection_summary", "")).strip(),
        "case_evidence_summary": str(parsed.get("case_evidence_summary", "")).strip(),
        "caution_notes": str(parsed.get("caution_notes", "")).strip(),
    }


def _result_output(result: Any) -> str:
    if hasattr(result, "output"):
        return str(result.output)
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def _feature_vector(features: dict[str, Any]) -> dict[str, float]:
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
    ]
    return {k: float(features.get(k, 0.0) or 0.0) for k in keys}


def _similar_cases(current_window: np.ndarray, cases: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    """Retrieve similar cases via comprehensive similarity on zscored windows."""
    if not cases:
        return []
    return retrieve_similar_cases(current_window, cases, top_k=top_k, window_key="zscored_window")


def _weights_from_cases(model_names: list[str], similar_cases: list[dict[str, Any]]) -> tuple[dict[str, float], str]:
    if not similar_cases:
        weights = {name: 0.0 for name in model_names}
        weights["LastClose"] = 0.55
        weights["RandomWalkDrift"] = 0.25
        remaining = [m for m in model_names if m not in {"LastClose", "RandomWalkDrift"}]
        for name in remaining:
            weights[name] = 0.20 / max(len(remaining), 1)
        return weights, "insufficient_case_evidence"

    scores = {name: 0.0 for name in model_names}
    for case in similar_cases:
        sim_w = float(case.get("similarity_weight", 1.0))
        metrics = case.get("metrics", {})
        for name in model_names:
            mse = float(metrics.get(name, {}).get("mse", np.inf))
            if np.isfinite(mse):
                scores[name] += sim_w / (mse + 1e-8)
    total = sum(scores.values())
    if total <= 0:
        return _weights_from_cases(model_names, [])
    weights = {name: float(score / total) for name, score in scores.items()}
    return weights, "similar_case_inverse_mse"


def _reference_prediction(baseline_predictions: dict[str, dict], weights: dict[str, float], horizon: int) -> list[float]:
    ref = np.zeros(horizon, dtype=float)
    for name, weight in weights.items():
        pred = np.asarray(baseline_predictions[name]["predictions"], dtype=float)
        ref += float(weight) * pred
    ref = np.where(np.isfinite(ref), ref, 1e-6)
    ref = np.maximum(ref, 1e-6)
    return [float(v) for v in ref.tolist()]


def _cluster_reference(
    clusters: list[dict[str, Any]],
    current_window: np.ndarray,
    baseline_predictions: dict[str, dict],
    horizon: int,
) -> list[float] | None:
    """Generate reference prediction from cluster-weighted model voting.

    Matches the current window to the best cluster via comprehensive similarity,
    then uses that cluster's model weight distribution to produce a weighted
    average prediction — mirroring AlphaCast's approach.
    """
    if not clusters:
        return None
    best = match_cluster(current_window, clusters)
    if best is None:
        return None
    model_weights = best.get("model_weights", {})
    total_weight = best.get("total_weight", 0)
    if not model_weights or total_weight <= 0:
        return None
    ref = np.zeros(horizon, dtype=float)
    for model_name, weight in model_weights.items():
        forecast = baseline_predictions.get(model_name)
        if forecast is None:
            continue
        pred = np.asarray(forecast["predictions"], dtype=float)
        ref += (float(weight) / float(total_weight)) * pred
    ref = np.where(np.isfinite(ref), ref, 1e-6)
    ref = np.maximum(ref, 1e-6)
    return [float(v) for v in ref.tolist()]


def _case_future_summary(case: dict[str, Any]) -> dict[str, Any]:
    truth = np.asarray(case.get("truth") or [], dtype=float)
    baseline_predictions = case.get("baseline_predictions") or {}
    last_close = np.nan
    try:
        last_close = float(baseline_predictions["LastClose"]["predictions"][0])
    except Exception:
        pass
    future_return = 0.0
    future_direction = "flat"
    if truth.size and np.isfinite(last_close) and last_close > 0 and truth[-1] > 0:
        future_return = float(np.log(float(truth[-1]) / last_close))
        if future_return > 1e-6:
            future_direction = "up"
        elif future_return < -1e-6:
            future_direction = "down"
    metrics = case.get("metrics") or {}
    ranked = sorted(
        (
            (name, float(values.get("mse", np.inf)))
            for name, values in metrics.items()
            if isinstance(values, dict) and np.isfinite(float(values.get("mse", np.inf)))
        ),
        key=lambda item: item[1],
    )
    return {
        "historical_future_return": future_return,
        "historical_future_direction": future_direction,
        "top_models_by_mse": [
            {"model": name, "mse": mse}
            for name, mse in ranked[:3]
        ],
    }


def _read_window_from_packet(packet: dict[str, Any]) -> pd.DataFrame:
    hist_dates = pd.to_datetime(packet.get("look_back_timestamps", []))
    if len(hist_dates) != len(packet["target_history"]):
        hist_dates = pd.date_range(packet["look_back_start"], packet["look_back_end"], periods=len(packet["target_history"]))
    data = {"date": hist_dates, packet["target_name"]: packet["target_history"]}
    for col, values in packet.get("exogenous_history", {}).items():
        data[col] = values
    return pd.DataFrame(data)


def build_baseline_packet(
    investigator_packet: dict[str, Any],
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    top_k: int = 20,
    use_llm: bool = False,
    precomputed_baseline_predictions: dict[str, dict] | None = None,
    case_records: list[dict[str, Any]] | None = None,
    training_case_offsets: set[int] | list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Pure prediction phase: load pre-built case library, run only
    cluster-voted models, blend cluster + case weights into reference.

    Requires analyze_training() to have been run first for this dataset.
    Mirrors AlphaCast's separation: training is one-time, prediction reads
    artifacts and only computes what's needed for the current window.
    """
    warnings: list[str] = []
    dataset = investigator_packet["dataset"]
    horizon = int(investigator_packet["forecast_horizon"])
    loader = FinCastDataLoader(manifest_path)
    entry = loader.resolve_dataset(dataset)
    target_history = np.asarray(investigator_packet.get("target_history", []), dtype=float)

    # --- Load pre-built case library + clusters (training artifacts) ---
    if case_records is None:
        _, cases, clusters = load_case_library(dataset, manifest_path)
    else:
        cases = case_records
        clusters = []

    # --- Match cluster, determine which models to run ---
    best_cluster = match_cluster(target_history, clusters) if clusters else None
    cluster_model_weights: dict[str, float] = {}
    if best_cluster is not None:
        cw = best_cluster.get("model_weights", {})
        tw = best_cluster.get("total_weight", 1)
        cluster_model_weights = {
            name: float(w) / float(tw) for name, w in cw.items()
        } if tw > 0 else {}

    # --- Run only cluster-voted models on current window ---
    current_window_df = _read_window_from_packet(investigator_packet)
    if precomputed_baseline_predictions is not None:
        baseline_predictions = precomputed_baseline_predictions
    elif cluster_model_weights:
        voted_names = list(cluster_model_weights.keys())
        all_models = {m.name: m for m in get_baseline_models()}
        baseline_predictions = {}
        for name in voted_names:
            model = all_models.get(name)
            if model is None:
                continue
            from fincast.basemodels.models import _safe_forecast
            fc = _safe_forecast(model, current_window_df, horizon,
                               investigator_packet["prediction_timestamps"])
            baseline_predictions[name] = fc.to_dict()
    else:
        # No cluster — run all 12 as fallback
        baseline_predictions = run_all_baselines(
            current_window_df, horizon, investigator_packet["prediction_timestamps"])

    model_names = list(baseline_predictions.keys())

    # --- Retrieve similar cases for evidence + case-based weighting ---
    current_end = pd.Timestamp(investigator_packet["look_back_end"])
    train_offsets = None if training_case_offsets is None else {int(v) for v in training_case_offsets}
    allowed_cases = [
        case for case in cases
        if pd.Timestamp(case["future_end"]) <= current_end
        and (train_offsets is None or int(case.get("window_offset", -1)) in train_offsets)
    ]
    if len(allowed_cases) < 5:
        warnings.append(f"Only {len(allowed_cases)} temporally valid cases available; using conservative fallback weights.")

    similar = _similar_cases(target_history, allowed_cases, top_k=top_k)
    weights, weighting_method = _weights_from_cases(model_names, similar)
    weight_sum = sum(weights.values()) or 1.0
    weights = {name: float(value / weight_sum) for name, value in weights.items()}

    # --- Neighbor-anchored calibration ---
    # Use top-1 similar case's truth as the neighbor magnitude anchor
    neighbor_magnitude = 0.0
    if similar:
        top1_truth = similar[0].get("truth")
        if top1_truth:
            nt = np.asarray(top1_truth, dtype=float)
            if nt.size >= 2 and nt[0] > 0:
                neighbor_magnitude = abs(float(nt[-1] / nt[0] - 1))

    # --- Compute references with adaptive cluster/case weighting ---
    case_reference = _reference_prediction(baseline_predictions, weights, horizon)
    cluster_ref = _cluster_reference(clusters, target_history, baseline_predictions, horizon)
    if cluster_ref is not None:
        cluster_sim = best_cluster.get("match_similarity", 0.5) if best_cluster else 0.0
        if cluster_sim >= 0.80:
            w_cluster = 0.70
        elif cluster_sim >= 0.50:
            w_cluster = 0.60
        else:
            w_cluster = 0.40
        w_case = 1.0 - w_cluster
        reference = [float(w_cluster * cr + w_case * car) for cr, car in zip(cluster_ref, case_reference)]
    else:
        reference = case_reference
        cluster_sim = 0.0
        w_cluster = 0.0
        w_case = 1.0

    # ── Extreme price regime detection ──
    # When current price is far outside the training distribution,
    # all models become unreliable. Fall back to HistoricAverage.
    lc_val = investigator_packet.get("financial_features", {}).get("last_close", 0.0)
    last_close = float(lc_val) if lc_val else 0.0
    price_extreme = False
    if cases and last_close > 0:
        all_window_means = []
        # Random sample across full time range (not just earliest windows)
        rng = np.random.default_rng(42)
        sample_cases = rng.choice(cases, size=min(200, len(cases)), replace=False)
        for c in sample_cases:
            rw = c.get("raw_window")
            if rw:
                all_window_means.append(np.mean(np.asarray(rw, dtype=float)))
        if all_window_means:
            case_mean = np.mean(all_window_means)
            case_std = np.std(all_window_means)
            if case_std > 0:
                z = abs(last_close - case_mean) / case_std
                if z > 3.0:
                    price_extreme = True
                    # Use HistoricAverage as a safe fallback
                    ha_fc = baseline_predictions.get("HistoricAverage")
                    if ha_fc:
                        ha_pred = np.asarray(ha_fc['predictions'], dtype=float)
                        reference = [float(v) for v in ha_pred.tolist()]
                        warnings.append(
                            f"Price extreme (z={z:.1f}): last_close=${last_close:.0f} "
                            f"vs case mean=${case_mean:.0f}. Using HistoricAverage fallback.")

    # ── Neighbor-anchored reference rescaling ──
    # Only activate when: (a) neighbor moved significantly, AND
    # (b) top-3 similar cases agree on direction (reduces regime-change errors)
    top3_dirs = [c.get("historical_future_direction", "neutral") for c in similar[:3]]
    dir_consensus = (
        top3_dirs.count("up") >= 2 or top3_dirs.count("down") >= 2
    ) if len(top3_dirs) >= 3 else False

    # ── Adaptive shrinkage to HistoricAverage ──
    # Only for high-volatility stocks (NFLX) where mean reversion dominates.
    vol = float(investigator_packet.get("financial_features", {}).get("realized_volatility_daily", 0.01) or 0.01)
    high_vol = vol > 0.025
    top1_sim = float(similar[0].get("similarity_score", 0.0) or 0.0) if (similar and similar[0].get("similarity_score") is not None) else 0.0

    if not price_extreme and high_vol:
        ha_fc = baseline_predictions.get("HistoricAverage")
        if ha_fc:
            ha_pred = np.asarray(ha_fc['predictions'], dtype=float)
            base_shrink = min(0.15, max(0.02, vol * 2.5))
            if top1_sim > 0.85:
                shrink = 0.0
            elif top1_sim > 0.75:
                shrink = base_shrink * 0.4
            else:
                shrink = min(base_shrink * 1.5, 0.30)
            reference = [float((1-shrink)*r + shrink*h) for r, h in zip(reference, ha_pred)]

    if (neighbor_magnitude > 0.08 and dir_consensus and high_vol
            and len(baseline_predictions) >= 6 and not price_extreme):
        # Filter: keep models whose predicted magnitude >= 20% of neighbor's
        min_mag = neighbor_magnitude * 0.20
        directional_preds = {}
        for name, fc in baseline_predictions.items():
            preds = np.asarray(fc['predictions'], dtype=float)
            if preds[0] > 0:
                mag = abs(float(preds[-1] / preds[0] - 1))
                if mag >= min_mag:
                    directional_preds[name] = fc

        if len(directional_preds) >= 3:
            dref = np.zeros(horizon, dtype=float)
            dw = 0.0
            for name, fc in directional_preds.items():
                w = weights.get(name, 1.0 / len(directional_preds))
                dref += w * np.asarray(fc['predictions'], dtype=float)
                dw += w
            if dw > 0:
                dref /= dw
                dref = np.maximum(dref, 1e-6)
                dref_list = [float(v) for v in dref.tolist()]

                # Safety valve: if the directional ref is too extreme (>50% change
                # or >3x the original reference), skip anchoring to avoid
                # catastrophic failures during regime-change windows.
                dref_mag = abs(float(dref_list[-1] / dref_list[0] - 1))
                ref_mag = abs(float(reference[-1] / reference[0] - 1)) if reference[0] > 0 else 0.0
                if dref_mag < 0.50 and (ref_mag == 0 or dref_mag < ref_mag * 3.0):
                    nconf = min(0.70, neighbor_magnitude * 3.0)
                    reference = [float((1-nconf)*r + nconf*d)
                                for r, d in zip(reference, dref_list)]
                    warnings.append(
                        f"Neighbor-anchored: {len(directional_preds)} dir models, "
                        f"mag={neighbor_magnitude*100:.0f}%, cases={top3_dirs}, blend={nconf:.2f}")
                else:
                    warnings.append(
                        f"Neighbor-anchored SKIPPED: directional ref magnitude "
                        f"({dref_mag*100:.0f}%) exceeds safety limit")

    similar_public = []
    for c in similar[:10]:
        public_case = {
            "window_offset": c["window_offset"],
            "look_back_end": c["look_back_end"],
            "future_end": c["future_end"],
            "best_model": c["best_model"],
            "similarity_score": c.get("similarity_score", 0.0),
            "similarity_weight": c.get("similarity_weight", 1.0),
        }
        public_case.update(_case_future_summary(c))
        similar_public.append(public_case)

    # Extract top-1 neighbor: the most similar historical window's actual future
    neighbor_truth = None
    neighbor_lookback = None
    if similar:
        top1 = similar[0]
        neighbor_truth = top1.get("truth")
        neighbor_lookback = top1.get("raw_window")

    packet = {
        "dataset": dataset,
        "ticker": investigator_packet["ticker"],
        "window_offset": investigator_packet["window_offset"],
        "forecast_horizon": horizon,
        "prediction_timestamps": investigator_packet["prediction_timestamps"],
        "baseline_predictions": baseline_predictions,
        "selected_models": [name for name, weight in sorted(weights.items(), key=lambda kv: kv[1], reverse=True) if weight > 1e-6],
        "model_weights": weights,
        "reference_prediction": reference,
        "reference_locked": price_extreme,
        "similar_cases": similar_public,
        "neighbor_truth": neighbor_truth,
        "neighbor_lookback": neighbor_lookback,
        "baseline_diagnostics": {
            "weighting_method": weighting_method,
            "allowed_case_count": len(allowed_cases),
            "similar_case_count": len(similar),
            "similarity_method": "comprehensive_similarity_5dim",
            "cluster_count": len(clusters),
            "cluster_reference_used": cluster_ref is not None,
            "cluster_match_similarity": float(cluster_sim),
            "cluster_weight": float(w_cluster),
            "case_weight": float(w_case),
            "cluster_voted_models": list(cluster_model_weights.keys()),
            "models_run": model_names,
            "case_filter": "case.future_end <= current.look_back_end",
            "case_split_filter": "all_cases" if train_offsets is None else "training_case_offsets_only",
            "case_cache_dataset": entry.name,
        },
        "llm_explanation": {
            "llm_explanation_available": False,
            "model_selection_summary": "",
            "case_evidence_summary": "",
            "caution_notes": "",
        },
        "warnings": warnings,
    }

    if use_llm:
        packet = attach_llm_explanation(packet)
    return packet


def build_baseline_agent(model_name: str | None = None):
    try:
        from pydantic_ai import Agent  # type: ignore
    except Exception as exc:
        raise RuntimeError("pydantic_ai is required for the LLM Baseline Agent explanation.") from exc
    resolved = model_name or _model_name_from_env()
    if not resolved:
        raise RuntimeError("No pydantic-ai model configured. Set PYA_MODEL or MODEL in FinCast/.env.")
    return Agent(resolved, instructions=BASELINE_AGENT_PROMPT)


def attach_llm_explanation(packet: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    try:
        agent = build_baseline_agent(model_name)
        compact = {
            "dataset": packet["dataset"],
            "ticker": packet["ticker"],
            "forecast_horizon": packet["forecast_horizon"],
            "model_weights": packet["model_weights"],
            "selected_models": packet["selected_models"],
            "similar_cases": packet["similar_cases"][:5],
            "baseline_warnings": packet["warnings"],
        }
        result = agent.run_sync(json.dumps(compact, ensure_ascii=False))
        parsed = _parse_jsonish(_result_output(result))
        parsed["llm_explanation_available"] = True
        packet["llm_explanation"] = parsed
    except Exception as exc:
        packet.setdefault("warnings", []).append(f"LLM baseline explanation unavailable: {exc}")
    return packet


def _result_output(result: Any) -> str:
    if hasattr(result, "output"):
        return str(result.output)
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def main() -> None:
    from fincast.tools import gather_forecast_inputs

    dataset = os.getenv("FINCAST_DATASET", "FinCastPrice_NVDA")
    offset = int(os.getenv("FINCAST_WINDOW_OFFSET", "1500"))
    horizon = int(os.getenv("FINCAST_FORECAST_HORIZON", "5"))
    use_llm = os.getenv("FINCAST_USE_LLM", "0") == "1"
    inv = gather_forecast_inputs(dataset, offset, horizon)
    packet = build_baseline_packet(inv, use_llm=use_llm)
    print(json.dumps(packet, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_AGENT_PROMPT",
    "attach_llm_explanation",
    "build_baseline_agent",
    "build_baseline_packet",
]
