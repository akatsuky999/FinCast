from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd

from fincast.tools.utils import (
    extract_dates_from_text,
    finite_array,
    historical_return_diagnostics,
    json_default,
    price_log_returns,
    safe_float,
)


REFLECTOR_AGENT_PROMPT = dedent(
    """
    You are ReflectorAgent for a time-series forecasting system.

    Audit the GeneratorAgent output using deterministic checks.
    You must approve only forecasts that:
      - have the requested length and timestamps,
      - are finite predictions,
      - do not jump beyond historical extreme step-change bounds,
      - do not cite future data or future dates,
      - justify meaningful adjustments using contextual evidence, similar cases, or model diagnostics,
      - do not confuse incremental-change forecasts with level forecasts.

    Return JSON with approved, issues, warnings, notes, and diagnostics.
    """
).strip()


def _text_blob(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=json_default)
    except Exception:
        return str(value)


def _timestamp_list(packet: dict[str, Any], key: str = "prediction_timestamps") -> list[str]:
    return [str(item) for item in packet.get(key, [])]


def _adjustment_has_support(generator_packet: dict[str, Any]) -> bool:
    reason = generator_packet.get("adjustment_reason", {})
    evidence = reason.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence_text = " ".join(str(item).lower() for item in evidence)
    diagnostics = generator_packet.get("generator_diagnostics", {})
    news = diagnostics.get("news_signal", {})
    cases = diagnostics.get("similar_case_signal", {})
    disagreement = diagnostics.get("baseline_disagreement", {})
    if evidence and any(token in evidence_text for token in ("news", "headline", "case", "similar", "model", "baseline", "arimax", "garch", "volatility")):
        return True
    if news.get("label") in {"positive", "negative"}:
        return True
    if safe_float(cases.get("support_strength"), 0.0) >= 0.58:
        return True
    if safe_float(disagreement.get("mean_relative_std"), 0.0) >= 0.02:
        return True
    return False


_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)(?:\s*%)?")


def _collect_numbers(value: Any, out: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = safe_float(value, np.nan)
        if np.isfinite(val):
            out.append(float(val))
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_numbers(item, out)


def _looks_like_date_fragment(text: str, start: int, end: int) -> bool:
    """Check if a numeric match is part of a date pattern."""
    before = text[start - 1 : start] if start > 0 else ""
    after = text[end : end + 1] if end < len(text) else ""
    after_next_digit = end + 1 < len(text) and text[end + 1].isdigit()
    before_prev_digit = start - 2 >= 0 and text[start - 2].isdigit()
    if (after == "-" and after_next_digit) or (before == "-" and before_prev_digit):
        return True
    if (after == ":" and after_next_digit) or (before == ":" and before_prev_digit):
        return True
    date_slice = text[start : min(len(text), start + 10)]
    if re.match(r"\d{4}-\d{2}-\d{2}", date_slice):
        return True
    return False


def _unsupported_numeric_claims(text: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    context_numbers: list[float] = []
    _collect_numbers(context, context_numbers)
    context_numbers = [v for v in context_numbers if np.isfinite(v)]
    unsupported: list[dict[str, Any]] = []
    if not text.strip() or not context_numbers:
        return unsupported

    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group(0).strip()
        start, end = match.span()
        before = text[max(0, start - 1) : start]
        after = text[end : min(len(text), end + 1)]
        if before == "-" or after in {"-", ":"}:
            continue
        # Skip numbers that look like date fragments (YYYY-MM-DD components)
        if _looks_like_date_fragment(text, start, end):
            continue
        try:
            value = float(match.group(1))
        except Exception:
            continue
        candidates = [value]
        if "%" in raw:
            candidates.append(value / 100.0)
        supported = False
        for candidate in candidates:
            for ctx in context_numbers:
                tolerance = max(0.03 * max(abs(candidate), abs(ctx), 1.0), 1e-4)
                if abs(candidate - ctx) <= tolerance:
                    supported = True
                    break
            if supported:
                break
        if not supported:
            snippet = text[max(0, start - 50) : min(len(text), end + 50)].strip()
            unsupported.append({"raw": raw, "value": value, "snippet": snippet})
        if len(unsupported) >= 8:
            break
    return unsupported


def reflect_forecast(
    generator_packet: dict[str, Any],
    investigator_packet: dict[str, Any] | None = None,
    baseline_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    investigator_packet = investigator_packet or {}
    baseline_packet = baseline_packet or {}
    issues: list[str] = []
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {}

    horizon = int(generator_packet.get("forecast_horizon") or baseline_packet.get("forecast_horizon") or 0)
    prediction_timestamps = _timestamp_list(generator_packet)
    expected_timestamps = _timestamp_list(baseline_packet) or _timestamp_list(investigator_packet)
    predictions = finite_array(generator_packet.get("final_prediction", []), fallback=np.nan)
    reference = finite_array(
        generator_packet.get("reference_prediction") or baseline_packet.get("reference_prediction", []),
        fallback=np.nan,
    )
    last_close = safe_float(
        investigator_packet.get("financial_features", {}).get("last_close"),
        safe_float(generator_packet.get("generator_diagnostics", {}).get("last_close"), 0.0),
    )
    if last_close <= 0:
        history = finite_array(investigator_packet.get("target_history", []), fallback=np.nan)
        if history.size:
            last_close = safe_float(history[-1], 0.0)

    if horizon <= 0:
        issues.append("forecast_horizon is missing or non-positive.")
    if predictions.size != horizon:
        issues.append(f"Prediction length {predictions.size} does not match forecast_horizon {horizon}.")
    if expected_timestamps and prediction_timestamps != expected_timestamps:
        issues.append("Prediction timestamps do not align with the baseline/investigator timestamps.")
    if len(prediction_timestamps) != horizon:
        issues.append("prediction_timestamps length does not match forecast_horizon.")
    if not predictions.size or not np.isfinite(predictions).all():
        issues.append("Predictions contain missing or non-finite values.")
    if predictions.size and last_close > 0:
        mean_pred = float(np.nanmean(predictions))
        scale_ratio = mean_pred / last_close
        diagnostics["scale_ratio_vs_last_value"] = scale_ratio
        if scale_ratio < 0.10 or scale_ratio > 10.0:
            issues.append("Prediction scale is implausible relative to the last observed value.")

        history = investigator_packet.get("target_history", [])
        hist_diag = historical_return_diagnostics(history)
        step_returns = price_log_returns(last_close, predictions)
        bound = max(hist_diag["abs_return_q995"] * 1.25, hist_diag["daily_volatility"] * 4.0, 1e-4)
        bound = min(bound, max(hist_diag["max_abs_return"] * 1.5, bound), 0.25)
        max_step = float(np.max(np.abs(step_returns))) if step_returns.size else 0.0
        diagnostics["max_forecast_step_abs_log_return"] = max_step
        diagnostics["historical_extreme_step_bound"] = float(bound)
        if max_step > bound + 1e-8:
            issues.append("Forecast jump exceeds the historical extreme return bound.")
        elif max_step > max(hist_diag["abs_return_q99"], hist_diag["daily_volatility"] * 3.0, 1e-4):
            warnings.append("Forecast contains a large step relative to recent historical returns.")

    # NEW: Detect if LLM just copied a single model without synthesizing
    if predictions.size and baseline_packet:
        single_model_copies = []
        for model_name, forecast in baseline_packet.get("baseline_predictions", {}).items():
            model_pred = finite_array(forecast.get("predictions", []), fallback=np.nan)
            if model_pred.size == predictions.size:
                if np.allclose(predictions, model_pred, rtol=1e-4):
                    single_model_copies.append(model_name)
        if single_model_copies:
            warnings.append(
                f"Prediction is an exact copy of {single_model_copies[0]} model output. "
                "LLM should synthesize across models, not delegate to a single one."
            )

    # NEW: Volatility-aware deviation check
    if predictions.size and reference.size == predictions.size and last_close > 0:
        garch_vol = safe_float(
            generator_packet.get("garch_volatility", {}).get("mean_daily_volatility"), 0.0
        )
        if garch_vol >= 0.05:
            deviation = float(np.mean(np.abs(predictions - reference)) / max(last_close, 1e-6))
            if deviation > 0.05:
                warnings.append(
                    f"High GARCH volatility ({garch_vol:.3f}) with large deviation "
                    f"from reference ({deviation:.3f}). Verify this is justified."
                )

    if reference.size == predictions.size and predictions.size and last_close > 0:
        mean_adjustment = float(np.mean(np.abs(predictions - reference)) / max(last_close, 1e-6))
        diagnostics["mean_relative_adjustment_vs_reference"] = mean_adjustment
        history_vol = historical_return_diagnostics(investigator_packet.get("target_history", [])).get("daily_volatility", 0.0)
        material_threshold = max(0.0025, history_vol * 0.25)
        diagnostics["material_adjustment_threshold"] = float(material_threshold)
        if mean_adjustment > material_threshold and not _adjustment_has_support(generator_packet):
            issues.append("Material adjustment from reference_prediction lacks news, similar-case, or model evidence.")
        elif mean_adjustment > 3.0 * material_threshold:
            # Large adjustment: require proportionally stronger evidence
            if not _adjustment_has_support(generator_packet):
                issues.append(
                    f"Large adjustment ({mean_adjustment:.4f}) from reference requires "
                    "strong evidence support."
                )
            else:
                warnings.append(
                    f"Large adjustment ({mean_adjustment:.4f}) from reference. "
                    "Verify this is justified by strong, multi-source evidence."
                )

    look_back_end_raw = investigator_packet.get("look_back_end")
    if look_back_end_raw:
        look_back_end = pd.Timestamp(look_back_end_raw)
        latest_news_date = investigator_packet.get("news_context", {}).get("latest_news_date")
        if latest_news_date and pd.Timestamp(latest_news_date) > look_back_end:
            issues.append("Investigator packet contains news dated after look_back_end.")
        reasoning_text = _text_blob(generator_packet.get("adjustment_reason", {})) + " " + _text_blob(generator_packet.get("llm_adjustment", {}))
        # Collect prediction timestamps to exclude from "future date" check
        known_timestamps = set()
        for ts_list_key in ("prediction_timestamps", "look_back_timestamps"):
            for ts_str in generator_packet.get(ts_list_key, []):
                try:
                    known_timestamps.add(pd.Timestamp(ts_str).strftime("%Y-%m-%d"))
                except Exception:
                    pass
        for ts_str in investigator_packet.get("prediction_timestamps", []):
            try:
                known_timestamps.add(pd.Timestamp(ts_str).strftime("%Y-%m-%d"))
            except Exception:
                pass

        future_dates = []
        for raw_date in extract_dates_from_text(reasoning_text):
            try:
                ts = pd.Timestamp(raw_date)
            except Exception:
                continue
            # Skip dates that are known prediction timestamps
            if ts.strftime("%Y-%m-%d") in known_timestamps:
                continue
            if ts > look_back_end:
                future_dates.append(raw_date)
        if future_dates:
            issues.append(f"Generator reasoning cites future date(s) beyond look_back_end: {future_dates[:3]}.")
        diagnostics["look_back_end"] = look_back_end.isoformat()

    numeric_context = {
        "generator": generator_packet,
        "investigator_financial_features": investigator_packet.get("financial_features", {}),
        "baseline": baseline_packet,
    }
    reasoning_text = _text_blob(generator_packet.get("adjustment_reason", {})) + " " + _text_blob(generator_packet.get("llm_adjustment", {}))
    unsupported_numbers = _unsupported_numeric_claims(reasoning_text, numeric_context)
    diagnostics["unsupported_numeric_claims"] = unsupported_numbers
    if unsupported_numbers:
        samples = ", ".join(item["raw"] for item in unsupported_numbers[:3])
        # LLM agent mode: semantic reasoning naturally produces interpretive numbers
        # (step indices, rounded percentages, derived stats). Treat as warnings,
        # not rejection issues — prediction values are validated separately.
        if generator_packet.get("generator_mode") == "llm_agent":
            warnings.append(f"Reasoning contains numeric claim(s) not found in context: {samples}. "
                            "Expected — LLM reasoning naturally uses approximate/derived numbers.")
        else:
            issues.append(f"Generator reasoning contains unsupported numeric claim(s): {samples}.")


    confidence = safe_float(generator_packet.get("confidence"), 0.0)
    disagreement = safe_float(generator_packet.get("generator_diagnostics", {}).get("baseline_disagreement", {}).get("mean_relative_std"), 0.0)
    garch_vol = safe_float(generator_packet.get("garch_volatility", {}).get("mean_daily_volatility"), 0.0)
    diagnostics["confidence"] = confidence
    diagnostics["baseline_disagreement_mean_relative_std"] = disagreement
    diagnostics["garch_mean_daily_volatility"] = garch_vol
    if disagreement >= 0.04 and confidence > 0.65:
        warnings.append("Confidence is high despite large baseline disagreement.")
    if garch_vol >= 0.05 and confidence > 0.70:
        warnings.append("Confidence is high despite high GARCH volatility.")

    approved = not issues
    notes = "Approved after financial sanity checks." if approved else "Rejected by financial sanity checks."
    report = {
        "approved": approved,
        "issues": issues,
        "warnings": warnings,
        "notes": notes,
        "diagnostics": diagnostics,
    }
    return json.loads(json.dumps(report, ensure_ascii=False, default=json_default, allow_nan=False))


def build_reflector_agent():
    try:
        from pydantic_ai import Agent  # type: ignore
        from pydantic_ai.messages import ModelResponse, TextPart  # type: ignore
        from pydantic_ai.models.function import FunctionModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("pydantic_ai is required to wrap Reflector as an Agent.") from exc

    def _extract_payload(messages: list[Any]) -> dict[str, Any]:
        for message in reversed(messages):
            for part in reversed(getattr(message, "parts", []) or []):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except Exception:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
        return {}

    def _reflector_model(messages: list[Any], agent_info: Any) -> ModelResponse:
        del agent_info
        payload = _extract_payload(messages)
        report = reflect_forecast(
            payload.get("generator_packet", {}),
            payload.get("investigator_packet", {}),
            payload.get("baseline_packet", {}),
        )
        return ModelResponse(parts=[TextPart(json.dumps(report, ensure_ascii=False, default=json_default))], model_name="function:fincast-reflector")

    return Agent(FunctionModel(function=_reflector_model), instructions=REFLECTOR_AGENT_PROMPT)


def main() -> None:
    from fincast.Agents.baseline_agent import build_baseline_packet
    from fincast.Agents.generator_agent import build_generator_packet
    from fincast.Agents.investigator_agent import run_investigator

    inv = run_investigator("FinCastPrice_NVDA", 1500, 5, use_llm=False)
    base = build_baseline_packet(inv, use_llm=False)
    gen = build_generator_packet(inv, base, use_llm=False)
    report = reflect_forecast(gen, inv, base)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "REFLECTOR_AGENT_PROMPT",
    "build_reflector_agent",
    "reflect_forecast",
]
