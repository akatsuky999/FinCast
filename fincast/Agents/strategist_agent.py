from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np

from fincast.tools.dataloader import DEFAULT_MANIFEST_PATH
from fincast.tools.utils import (
    clip_price_path_by_return_bounds,
    finite_array,
    historical_return_diagnostics,
    json_default,
    lexical_news_signal,
    model_prediction_summary,
    parse_jsonish,
    safe_float,
    validate_price_prediction,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PACKAGE_ROOT / ".env"


def api_key_count() -> int:
    """Module-level: count available API keys from .env (no function call needed)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
        count = 1  # primary key
        for i in range(1, 20):
            if os.getenv(f"OPENAI_API_KEY_{i}", "").strip():
                count += 1
            else:
                break
        return count
    except Exception:
        return 1


STRATEGIST_AGENT_SYSTEM_PROMPT = dedent(
    """
You are the core forecasting agent in an agentic time-series prediction
system. You receive a reference forecast (mechanical ensemble of 12 models),
the most similar historical window and its actual outcome (neighbor), recent
news headlines, and case/model diagnostics.

Your unique role: the reference cannot read news, compare trajectory shapes,
or weigh conflicting evidence. YOU can. Design the forecast trajectory by:

1. READING the news headlines — assess qualitative impact. An analyst
   upgrade > generic "positive outlook." A product launch > minor partnership.
2. COMPARING neighbor_truth shape vs reference_prediction shape. If the
   neighbor peaked early then reversed but reference rises monotonically,
   reshape toward the neighbor pattern.
3. WEIGHING evidence: news + cases + models aligned → confident 3-5% move.
   Only 1 source has signal → <1% adjustment. Conflicting → stay close
   to reference.
4. CALL emit_prediction(predictions, reasoning, evidence_summary, risk_notes)
   exactly once. predictions: list of H finite floats.

Be decisive when evidence aligns. Be humble when it doesn't.
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


def _result_output(result: Any) -> str:
    if hasattr(result, "output"):
        return str(result.output)
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def _garch_volatility(baseline_packet: dict[str, Any]) -> dict[str, Any]:
    try:
        values = baseline_packet["baseline_predictions"]["ARMAGARCHReturn"]["metadata"]["variance_forecast"]
    except Exception:
        values = []
    variance = np.maximum(finite_array(values, fallback=0.0), 0.0)
    if variance.size == 0:
        return {"mean_daily_volatility": 0.0, "max_daily_volatility": 0.0, "variance_forecast": []}
    vol = np.sqrt(variance)
    return {
        "mean_daily_volatility": float(np.mean(vol)),
        "max_daily_volatility": float(np.max(vol)),
        "variance_forecast": [float(v) for v in variance.tolist()],
    }


def _baseline_disagreement(baseline_packet: dict[str, Any], reference: np.ndarray) -> dict[str, float]:
    rows = []
    for forecast in baseline_packet.get("baseline_predictions", {}).values():
        arr = finite_array(forecast.get("predictions", []), fallback=np.nan)
        if arr.size == reference.size and np.isfinite(arr).all():
            rows.append(arr)
    if len(rows) < 2:
        return {"mean_relative_std": 0.0, "max_relative_std": 0.0}
    mat = np.vstack(rows)
    denom = np.maximum(np.abs(reference), 1e-6)
    rel = np.std(mat, axis=0) / denom
    return {
        "mean_relative_std": float(np.mean(rel)),
        "max_relative_std": float(np.max(rel)),
    }


def _similar_case_signal(similar_cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not similar_cases:
        return {
            "weighted_future_return": 0.0,
            "up_weight": 0.0,
            "down_weight": 0.0,
            "dominant_direction": "neutral",
            "support_strength": 0.0,
        }
    up = 0.0
    down = 0.0
    total = 0.0
    weighted_return = 0.0
    for case in similar_cases:
        weight = safe_float(case.get("similarity_weight"), 1.0)
        ret = safe_float(case.get("historical_future_return"), 0.0)
        total += weight
        weighted_return += weight * ret
        if ret > 1e-6:
            up += weight
        elif ret < -1e-6:
            down += weight
    total = total or 1.0
    up_share = up / total
    down_share = down / total
    dominant = "neutral"
    support = max(up_share, down_share)
    if up_share >= 0.58:
        dominant = "up"
    elif down_share >= 0.58:
        dominant = "down"
    else:
        support = abs(up_share - down_share)
    return {
        "weighted_future_return": float(weighted_return / total),
        "up_weight": float(up_share),
        "down_weight": float(down_share),
        "dominant_direction": dominant,
        "support_strength": float(np.clip(support, 0.0, 1.0)),
    }


def _all_recent_headlines(briefing_packet: dict[str, Any]) -> list[str]:
    news = briefing_packet.get("news_context", {})
    headlines: list[str] = []
    for key in ("recent_1d", "recent_3d", "recent_5d", "recent_20d"):
        bucket = news.get(key) or {}
        headlines.extend(str(item) for item in bucket.get("top_headlines", []) if item)
    headlines.extend(str(item) for item in news.get("top_headlines", []) if item)
    deduped = []
    seen = set()
    for item in headlines:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped[:20]


def _build_rich_context(briefing_packet: dict[str, Any], baseline_packet: dict[str, Any]) -> dict[str, Any]:
    """Build a streamlined context packet for the LLM agent.

    Only includes information the LLM needs for semantic reasoning —
    structured diagnostics, not verbose text summaries.
    """
    reference = finite_array(baseline_packet["reference_prediction"], fallback=1.0)
    last_close = safe_float(
        briefing_packet.get("financial_features", {}).get("last_close"),
        float(reference[0]) if reference.size else 1.0,
    )
    horizon = int(baseline_packet["forecast_horizon"])

    garch_vol = _garch_volatility(baseline_packet)
    disagreement = _baseline_disagreement(baseline_packet, reference)
    similar_signal = _similar_case_signal(baseline_packet.get("similar_cases", []))
    headlines = _all_recent_headlines(briefing_packet)
    news_signal = lexical_news_signal(headlines)
    hist_diag = historical_return_diagnostics(briefing_packet.get("target_history", []))

    # Streamlined model consensus — only key numbers, no verbose text
    mc = model_prediction_summary(
        baseline_packet["baseline_predictions"],
        baseline_packet.get("model_weights", {}),
        horizon,
        last_close,
    )

    # Streamlined similar cases — top 3 only, key fields
    similar_cases_text = []
    for i, case in enumerate(baseline_packet.get("similar_cases", [])[:3]):
        similar_cases_text.append({
            "sim_w": safe_float(case.get("similarity_weight"), 0.0),
            "dir": case.get("historical_future_direction", "?"),
            "ret_pct": round(safe_float(case.get("historical_future_return"), 0.0) * 100, 1),
        })

    # Neighbor summary
    neighbor_raw = baseline_packet.get("neighbor_lookback")
    neighbor_truth_raw = baseline_packet.get("neighbor_truth")
    neighbor_summary = None
    if neighbor_raw and neighbor_truth_raw:
        n_tr = np.asarray(neighbor_truth_raw, dtype=float)
        if n_tr.size >= 2:
            n_ret = float(np.log(n_tr[-1] / n_tr[0])) * 100
            neighbor_summary = {
                "ret_pct": round(n_ret, 1),
                "dir": "up" if n_ret > 0.5 else ("down" if n_ret < -0.5 else "flat"),
                "peak_step": int(np.argmax(n_tr)),
                "trough_step": int(np.argmin(n_tr)),
                "trajectory": f"{n_tr[0]:.1f}→peak{n_tr[int(np.argmax(n_tr))]:.1f}@step{int(np.argmax(n_tr))}→end{n_tr[-1]:.1f}",
            }

    # ── Evidence alignment ──
    ev_news_dir = "up" if news_signal["label"] == "positive" else ("down" if news_signal["label"] == "negative" else "neutral")
    ev_cases = similar_signal["dominant_direction"]
    ev_consensus = mc["consensus_direction"]
    aligned, conflicting = [], []
    if ev_news_dir == ev_cases and ev_news_dir != "neutral":
        aligned.append("news↔cases")
    elif ev_news_dir != "neutral" and ev_cases != "neutral":
        conflicting.append("news↔cases")
    if ev_cases == ev_consensus and ev_cases != "neutral":
        aligned.append("cases↔models")
    elif ev_cases != "neutral" and ev_consensus != "neutral":
        conflicting.append("cases↔models")

    # Streamlined context — only what the LLM needs for trajectory design
    return {
        "ticker": briefing_packet.get("ticker", ""),
        "horizon": horizon,
        "last_close": float(last_close),
        "look_back_end": briefing_packet.get("look_back_end"),
        "prediction_start": briefing_packet.get("prediction_start"),
        # Core forecast data
        "reference_prediction": [float(round(v, 3)) for v in reference.tolist()],
        # Model consensus: only direction + disagreement + price range
        "consensus_dir": mc["consensus_direction"],
        "disagreement": safe_float(disagreement.get("mean_relative_std"), 0),
        "garch_daily_vol": safe_float(garch_vol.get("mean_daily_volatility"), 0),
        "final_price_range": mc.get("final_price_range", {}),
        # Cases: top-3 + aggregated signal
        "similar_cases": similar_cases_text,
        "case_signal": {"dir": similar_signal["dominant_direction"],
                        "up": round(similar_signal["up_weight"], 2),
                        "down": round(similar_signal["down_weight"], 2),
                        "support": round(similar_signal["support_strength"], 2)},
        # Neighbor: summary stats + raw arrays for shape comparison
        "neighbor_summary": neighbor_summary,
        "neighbor_lookback": baseline_packet.get("neighbor_lookback"),
        "neighbor_truth": baseline_packet.get("neighbor_truth"),
        # News: raw headlines for semantic reading + quick label
        "headlines": headlines[:8],
        "news_label": news_signal["label"],
        "news_score": round(news_signal["score"], 2),
        # Evidence alignment
        "evidence": {"aligned": aligned, "conflicting": conflicting},
        # Diagnostics: only the 6 most informative
        "diag": {
            k: round(briefing_packet.get("financial_features", {}).get(k, 0), 4)
            for k in ["realized_volatility_daily", "max_drawdown", "skewness",
                       "trend_slope_log_price", "cumulative_log_return", "volume_spike_ratio_20d"]
        },
        # Historical bounds for clipping awareness
        "hist_bounds": {k: round(v, 4) for k, v in hist_diag.items()},
    }


def create_strategist_agent(model_name: str | None = None, manifest_path: str | Path = DEFAULT_MANIFEST_PATH):
    """Build the StrategistAgent as Investment Committee Chair with tools."""
    try:
        from pydantic_ai import Agent, RunContext
    except Exception as exc:
        raise RuntimeError(
            "pydantic_ai is required for the LLM Strategist Agent."
        ) from exc
    globals()["RunContext"] = RunContext

    resolved = model_name or _model_name_from_env()
    if not resolved:
        raise RuntimeError(
            "No pydantic-ai model configured. Set PYA_MODEL or MODEL in FinCast/.env."
        )

    agent = Agent(resolved, instructions=STRATEGIST_AGENT_SYSTEM_PROMPT)

    # Mutable container so run_strategist() can retrieve the structured prediction
    # directly, bypassing unreliable text-parsing of LLM output.
    agent._emitted_prediction: dict[str, Any] | None = None

    @agent.tool
    def consult(
        ctx: RunContext[None],
        dataset_name: str,
        window_offset: int = 0,
        forecast_horizon: int | None = None,
    ) -> dict[str, Any]:
        """Fetch the research briefing for a dataset window.

        Returns the reference prediction (already fused from 12 models via
        cluster-weighted voting and case-similarity blending), plus supporting
        evidence: model consensus, similar historical cases, news sentiment,
        volatility diagnostics, and historical step-change bounds.
        Call this FIRST in every forecasting step.
        """
        from fincast.Agents.baseline_agent import build_baseline_packet
        from fincast.Agents.briefing_agent import run_briefing

        inv = run_briefing(
            dataset_name=dataset_name,
            window_offset=int(window_offset),
            forecast_horizon=forecast_horizon,
            manifest_path=manifest_path,
            use_llm=False,
        )
        base = build_baseline_packet(inv, manifest_path=manifest_path, use_llm=False)
        return _build_rich_context(inv, base)

    @agent.tool
    def emit_prediction(
        ctx: RunContext[None],
        predictions: list[float],
        reasoning: str,
        evidence_summary: list[str],
        risk_notes: str,
    ) -> dict[str, Any]:
        """Submit your final price-level prediction.

        This is the terminal action. The prediction will be validated by the
        Reflector for financial sanity (positive prices, reasonable scale,
        within historical return bounds, evidence-supported adjustments).

        Args:
            predictions: List of exactly forecast_horizon positive price values
            reasoning: Concise summary of your analysis and decision rationale
            evidence_summary: Specific evidence items grounding your decision
            risk_notes: Key risks and uncertainties for this prediction
        """
        pred_arr = np.asarray(predictions, dtype=float)
        issues = []
        if pred_arr.size == 0:
            issues.append("Predictions list is empty")
        if not np.all(np.isfinite(pred_arr)):
            issues.append("Predictions contain non-finite values")
        if pred_arr.size and np.any(pred_arr <= 0):
            issues.append("Stock prices must be positive")
        if not reasoning.strip():
            issues.append("Reasoning is required; explain your analysis")
        if not evidence_summary:
            issues.append("At least one evidence item is required")

        if issues:
            rejected = {
                "accepted": False,
                "issues": issues,
                "hint": "Fix the issues above and call emit_prediction again with corrected values.",
            }
            agent._emitted_prediction = rejected
            return rejected

        accepted = {
            "accepted": True,
            "predictions": [float(v) for v in pred_arr.tolist()],
            "predictions_count": int(pred_arr.size),
            "reasoning": reasoning.strip(),
            "evidence_summary": evidence_summary,
            "risk_notes": risk_notes.strip(),
        }
        agent._emitted_prediction = accepted
        return accepted

    return agent


def _deterministic_fallback(
    briefing_packet: dict[str, Any],
    baseline_packet: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic strategist: evidence → direction × strength → adjustment.

    Unlike the old version that crushed every signal through heavy damping,
    this produces visible adjustments (1-3% cumulative for strong signals)
    so the case/news evidence pipeline actually matters.
    """
    warnings = list(briefing_packet.get("warnings", [])) + list(baseline_packet.get("warnings", []))
    reference = finite_array(baseline_packet["reference_prediction"], fallback=1.0)
    last_close = safe_float(
        briefing_packet.get("financial_features", {}).get("last_close"),
        float(reference[0]) if reference.size else 1.0,
    )
    horizon = int(baseline_packet["forecast_horizon"])

    garch_vol = _garch_volatility(baseline_packet)
    disagreement = _baseline_disagreement(baseline_packet, reference)
    similar_signal = _similar_case_signal(baseline_packet.get("similar_cases", []))
    headlines = _all_recent_headlines(briefing_packet)
    news_signal = lexical_news_signal(headlines)

    direction = "neutral"
    strength = 0.0
    evidence: list[str] = []

    # ── evidence → direction judgement ──
    if news_signal["label"] in {"positive", "negative"} and similar_signal["dominant_direction"] in {"up", "down"}:
        if news_signal["label"] == "positive" and similar_signal["dominant_direction"] == "up":
            direction = "up"
            strength = 0.30 + 0.35 * similar_signal["support_strength"]
            evidence.append("News sentiment and historical cases both point upward.")
        elif news_signal["label"] == "negative" and similar_signal["dominant_direction"] == "down":
            direction = "down"
            strength = 0.30 + 0.35 * similar_signal["support_strength"]
            evidence.append("News sentiment and historical cases both point downward.")
    elif similar_signal["dominant_direction"] in {"up", "down"} and similar_signal["support_strength"] >= 0.65:
        direction = similar_signal["dominant_direction"]
        strength = 0.20 + 0.25 * similar_signal["support_strength"]
        evidence.append("Historical cases strongly favor one direction.")
    elif news_signal["label"] in {"positive", "negative"}:
        direction = "up" if news_signal["label"] == "positive" else "down"
        strength = 0.15
        evidence.append("News sentiment provides a moderate directional signal.")
    # neighbor check: if the single most similar historical window moved sharply
    neighbor_truth = baseline_packet.get("neighbor_truth")
    if neighbor_truth and direction == "neutral" and last_close > 0:
        ntruth = np.asarray(neighbor_truth, dtype=float)
        if ntruth.size >= horizon:
            neighbor_return = float(np.log(ntruth[-1] / ntruth[0]))
            if abs(neighbor_return) > 0.05:
                direction = "up" if neighbor_return > 0 else "down"
                strength = 0.12
                evidence.append(
                    f"The most similar historical window moved {neighbor_return*100:+.1f}%; "
                    "weak signal from pattern similarity."
                )

    direction_sign = 1.0 if direction == "up" else (-1.0 if direction == "down" else 0.0)
    volatility = max(garch_vol.get("mean_daily_volatility", 0.0), safe_float(
        briefing_packet.get("financial_features", {}).get("realized_volatility_daily"), 0.0
    ))
    disagreement_val = safe_float(disagreement.get("mean_relative_std"), 0.0)

    hist_diag = historical_return_diagnostics(briefing_packet.get("target_history", []))
    # Per-step cap: adapts to stock volatility + neighbor magnitude
    base_cap = hist_diag["abs_return_q95"] * 2.0
    # Enrich with neighbor magnitude if available
    neighbor_cap = 0.0
    neighbor_truth_val = baseline_packet.get("neighbor_truth")
    if neighbor_truth_val:
        nt = np.asarray(neighbor_truth_val, dtype=float)
        if nt.size >= 2 and nt[0] > 0:
            neighbor_cap = abs(float(nt[-1] / nt[0] - 1)) / horizon
    adjustment_cap = max(base_cap, neighbor_cap * 0.5)
    adjustment_cap = min(adjustment_cap, max(0.08, volatility * 2.5))  # dynamic ceiling
    # Lighter damping — high vol means larger moves are EXPECTED
    vol_damp = 1.0 / (1.0 + 1.5 * volatility)
    dis_damp = 1.0 / (1.0 + 3.0 * disagreement_val)
    signed_adj = direction_sign * strength * adjustment_cap * vol_damp * dis_damp

    ramp = np.linspace(1.0 / horizon, 1.0, horizon) if horizon > 0 else np.array([1.0])
    adjusted = reference * np.exp(signed_adj * ramp)
    final_prediction, bound_diag = clip_price_path_by_return_bounds(
        adjusted, last_close,
        briefing_packet.get("target_history", []),
        multiplier=1.25,
    )

    # Confidence: higher when evidence is multi-source and disagreement is low
    confidence = 0.30 + 0.40 * strength
    confidence += 0.15 * (1.0 - min(disagreement_val / 0.05, 1.0))
    confidence += 0.15 * (1.0 - min(volatility / 0.04, 1.0))
    confidence = float(np.clip(confidence, 0.15, 0.85))

    return {
        "dataset": baseline_packet["dataset"],
        "ticker": baseline_packet.get("ticker", ""),
        "window_offset": baseline_packet["window_offset"],
        "forecast_horizon": horizon,
        "prediction_timestamps": baseline_packet.get("prediction_timestamps", []),
        "final_prediction": final_prediction,
        "strategist_mode": "deterministic_degraded",
        "adjustment_reason": {
            "policy": "deterministic_evidence_driven",
            "selected_direction": direction,
            "selected_strength": float(strength),
            "signed_log_return_adjustment": float(signed_adj),
            "cumulative_adjustment_pct": float(np.exp(signed_adj) - 1) * 100,
            "evidence": evidence,
            "risk_notes": "Degraded mode: LLM agent unavailable. Using formula-based directional adjustment. "
                          "This is a scalar direction×strength adjustment — it cannot reshape the forecast "
                          "trajectory based on neighbor pattern comparison. Results are inferior to LLM agent mode.",
        },
        "confidence": confidence * 0.80,  # Penalty for degraded mode
        "strategist_diagnostics": {
            "baseline_disagreement": disagreement,
            "similar_case_signal": similar_signal,
            "news_signal": news_signal,
            "return_bound_diagnostics": bound_diag,
        },
        "llm_adjustment": {
            "llm_adjustment_available": False,
            "source": "deterministic_fallback",
        },
        "warnings": warnings,
    }


def run_strategist(
    briefing_packet: dict[str, Any],
    baseline_packet: dict[str, Any],
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    use_llm: bool = True,
    model_name: str | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run the Strategist — the core forecasting agent.

    Primary mode (use_llm=True): LLM agent synthesizes neighbor patterns,
    case evidence, model consensus, and news signals to make shape-level
    trajectory adjustments. This is what makes the system "agentic."

    Degraded mode (use_llm=False): formula-based directional adjustment.
    Cannot reshape trajectories or weigh conflicting evidence. Produces
    inferior results. Only use when LLM is unavailable.

    The LLM path includes a retry loop: if the Reflector rejects the
    prediction, the issues are fed back to the LLM for correction.
    """
    # ── Reference locked mode: skip all adjustment ──
    if baseline_packet.get("reference_locked"):
        ref = finite_array(baseline_packet["reference_prediction"], fallback=1.0)
        ref_list = [float(v) for v in ref.tolist()]
        return {
            "dataset": baseline_packet["dataset"],
            "ticker": baseline_packet.get("ticker", ""),
            "window_offset": baseline_packet["window_offset"],
            "forecast_horizon": int(baseline_packet["forecast_horizon"]),
            "prediction_timestamps": baseline_packet.get("prediction_timestamps", []),
            "final_prediction": ref_list,
            "strategist_mode": "reference_locked",
            "adjustment_reason": {
                "policy": "reference_locked_extreme_price",
                "evidence": ["Reference locked: extreme price regime detected. No adjustment applied."],
                "risk_notes": "Price outside normal range — using safe fallback.",
            },
            "confidence": 0.25,
            "strategist_diagnostics": {},
            "llm_adjustment": {"llm_adjustment_available": False, "source": "locked"},
            "warnings": list(briefing_packet.get("warnings", [])) + list(baseline_packet.get("warnings", [])),
        }

    if not use_llm:
        return _deterministic_fallback(briefing_packet, baseline_packet)

    warnings = list(briefing_packet.get("warnings", [])) + list(baseline_packet.get("warnings", []))
    horizon = int(baseline_packet["forecast_horizon"])
    last_close = safe_float(
        briefing_packet.get("financial_features", {}).get("last_close"),
        float(finite_array(baseline_packet.get("reference_prediction", []), fallback=1.0)[0]),
    )

    # Build context once (shared across retries)
    context = _build_rich_context(briefing_packet, baseline_packet)

    # ── Raw API call (fast path) ──
    resolved_model = model_name or _model_name_from_env()
    try:
        from openai import OpenAI
        import httpx
        import threading

        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        # ── API Key pool for concurrent load distribution ──
        api_keys: list[str] = []
        primary = os.getenv("OPENAI_API_KEY", "").strip()
        if primary:
            api_keys.append(primary)
        # Scan OPENAI_API_KEY_1, OPENAI_API_KEY_2, ...
        for i in range(1, 20):
            extra = os.getenv(f"OPENAI_API_KEY_{i}", "").strip()
            if extra and extra not in api_keys:
                api_keys.append(extra)
            elif not extra:
                break

        # One httpx client per key (shared within same-key requests)
        _http_clients = [
            httpx.Client(
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
            for _ in api_keys
        ]
        client_pool = [
            OpenAI(api_key=k, base_url=base_url, timeout=120.0, max_retries=1, http_client=hc)
            for k, hc in zip(api_keys, _http_clients)
        ]
        _pool_lock = threading.Lock()
        _pool_idx = [0]

        def _get_client() -> OpenAI:
            """Round-robin across API keys for balanced concurrent load."""
            with _pool_lock:
                c = client_pool[_pool_idx[0] % len(client_pool)]
                _pool_idx[0] += 1
                return c

        client = client_pool[0]  # Default client (single-key compatibility)
        api_model = resolved_model.split(":", 1)[1] if ":" in (resolved_model or "") else (resolved_model or "deepseek-v4-flash")
        n_keys = len(api_keys)
    except Exception:
        client = None
        api_model = None
        n_keys = 0

    # Module-level diagnostics for external query before run_strategist() is called
    _key_pool_size = n_keys

    def _call_llm(prompt_text: str, feedback_text: str = "") -> dict[str, Any] | None:
        """Call LLM via raw API (fast) or pydantic-ai (fallback). Returns parsed prediction dict or None."""
        # Fast path: raw API, round-robin across key pool for concurrency
        if client is not None:
            try:
                full_prompt = prompt_text
                if feedback_text:
                    full_prompt += f"\n\n[REVISION REQUEST]\n{feedback_text}\nPlease fix and retry."
                c = _get_client() if n_keys > 1 else client
                resp = c.chat.completions.create(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": STRATEGIST_AGENT_SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt},
                    ],
                    max_tokens=2500,
                    temperature=0.3,
                )
                output = resp.choices[0].message.content or ""
                # Parse: look for emit_prediction call or JSON with predictions key
                import re
                # Try direct JSON parse first
                try:
                    parsed = json.loads(output)
                    if "predictions" in parsed and isinstance(parsed["predictions"], list):
                        return parsed
                except Exception:
                    pass
                # Try to extract from markdown code blocks
                for pattern in [r'```(?:json)?\s*\n?(.*?)\n?```', r'\{[^{}]*"predictions"\s*:\s*\[[^\]]*\][^{}]*\}']:
                    matches = re.findall(pattern, output, re.DOTALL)
                    for match in matches:
                        try:
                            parsed = json.loads(match.strip() if pattern.startswith(r'```') else match)
                            if "predictions" in parsed and isinstance(parsed["predictions"], list):
                                return parsed
                        except Exception:
                            continue
                # Last resort: find any {...} with predictions
                s = output.find('{"predictions"')
                if s >= 0:
                    e = output.find('}', output.rfind(']')) + 1
                    if e > s:
                        try:
                            parsed = json.loads(output[s:e])
                            if "predictions" in parsed:
                                return parsed
                        except Exception:
                            pass
                return None
            except Exception as exc:
                # Timeout/connection error — likely concurrency too high for API
                import sys
                err_type = type(exc).__name__
                hint = ""
                if "timeout" in err_type.lower():
                    hint = " (reduce test_concurrency)"
                elif "rate" in str(exc).lower() or "429" in str(exc):
                    hint = " (rate limited — reduce concurrency or add keys)"
                print(f"\n  [warn] API {err_type}{hint}", file=sys.stderr, flush=True)

        # Raw API failed — return None so run_strategist falls back to deterministic.
        # No pydantic-ai retry (it would also fail under the same API load).
        return None

    # ── Retry loop with raw API ──
    ctx_json = json.dumps(context, ensure_ascii=False, default=json_default)
    prompt = (
        f"Research briefing for {horizon}-step forecast. Analyze, then output a JSON "
        f"object with keys: predictions (list of {horizon} floats), reasoning (string), "
        f"evidence_summary (list of strings), risk_notes (string).\n\n"
        + ctx_json
    )

    feedback = ""
    prediction_result = None
    for attempt in range(max_retries + 1):
        try:
            raw = _call_llm(prompt, feedback)
            if raw is None:
                raise RuntimeError("LLM returned no valid prediction")
            # Normalize: accept raw API format directly
            if "predictions" in raw and isinstance(raw["predictions"], list):
                prediction_result = {
                    "accepted": True,
                    "predictions": raw["predictions"],
                    "reasoning": str(raw.get("reasoning", "")),
                    "evidence_summary": raw.get("evidence_summary", []) if isinstance(raw.get("evidence_summary"), list) else [],
                    "risk_notes": str(raw.get("risk_notes", "")),
                }
            else:
                prediction_result = raw  # Use as-is (pydantic-ai format)
        except Exception as exc:
            warnings.append(f"LLM call failed (attempt {attempt+1}): {exc}")
            if attempt < max_retries:
                feedback = f"LLM error: {exc}. Please retry."
                continue
            result = _deterministic_fallback(briefing_packet, baseline_packet)
            result.setdefault("warnings", []).append(f"LLM failed after {max_retries+1} attempts: {exc}")
            return result

        if prediction_result is None or not prediction_result.get("accepted"):
            issues = prediction_result.get("issues", ["No valid prediction emitted"]) if prediction_result else ["No prediction"]
            feedback = "; ".join(str(i) for i in issues)
            if attempt < max_retries:
                continue
            result = _deterministic_fallback(briefing_packet, baseline_packet)
            result.setdefault("warnings", []).append(f"LLM failed after {max_retries+1} attempts: {feedback}")
            return result

        break

    if prediction_result is None or not prediction_result.get("accepted"):
        result = _deterministic_fallback(briefing_packet, baseline_packet)
        result.setdefault("warnings", []).append("LLM did not produce valid prediction; using fallback.")
        return result

    # Build the final prediction from LLM output
    llm_predictions = np.asarray(prediction_result["predictions"], dtype=float)

    # Normalize length
    if llm_predictions.size != horizon:
        if llm_predictions.size > horizon:
            llm_predictions = llm_predictions[:horizon]
        else:
            last_val = float(llm_predictions[-1]) if llm_predictions.size > 0 else last_close
            padding = np.full(horizon - llm_predictions.size, last_val, dtype=float)
            llm_predictions = np.concatenate([llm_predictions, padding])

    # Apply safety clip as a post-hoc guard (not a formula override)
    final_prediction, bound_diag = clip_price_path_by_return_bounds(
        llm_predictions.tolist() if hasattr(llm_predictions, 'tolist') else list(llm_predictions),
        last_close,
        briefing_packet.get("target_history", []),
        multiplier=1.25,
    )

    # Validate
    validation = validate_price_prediction(
        final_prediction, last_close,
        briefing_packet.get("target_history", []),
        horizon,
    )
    if not validation["valid"]:
        warnings.append(f"Post-hoc validation issues: {'; '.join(validation['issues'])}")

    garch_vol = _garch_volatility(baseline_packet)
    disagreement = _baseline_disagreement(baseline_packet, finite_array(baseline_packet["reference_prediction"], fallback=last_close))
    similar_signal = _similar_case_signal(baseline_packet.get("similar_cases", []))
    headlines = _all_recent_headlines(briefing_packet)
    news_signal = lexical_news_signal(headlines)

    confidence = 0.55
    volatility = max(garch_vol.get("mean_daily_volatility", 0.0), safe_float(
        briefing_packet.get("financial_features", {}).get("realized_volatility_daily"), 0.0
    ))
    disagreement_val = safe_float(disagreement.get("mean_relative_std"), 0.0)
    confidence -= min(volatility / 0.05, 1.0) * 0.20
    confidence -= min(disagreement_val / 0.04, 1.0) * 0.15
    confidence = float(np.clip(confidence, 0.15, 0.85))

    return {
        "dataset": baseline_packet["dataset"],
        "ticker": baseline_packet.get("ticker", ""),
        "window_offset": baseline_packet["window_offset"],
        "forecast_horizon": horizon,
        "prediction_timestamps": baseline_packet.get("prediction_timestamps", []),
        "reference_prediction": baseline_packet.get("reference_prediction", []),
        "final_prediction": [float(v) for v in final_prediction],
        "strategist_mode": "llm_agent",
        "adjustment_reason": {
            "policy": "llm_agent_shape_level_reasoning",
            "evidence": prediction_result.get("evidence_summary", []),
            "risk_notes": prediction_result.get("risk_notes", ""),
        },
        "confidence": confidence,
        "strategist_diagnostics": {
            "baseline_disagreement": disagreement,
            "similar_case_signal": similar_signal,
            "news_signal": news_signal,
            "return_bound_diagnostics": bound_diag,
        },
        "llm_adjustment": {
            "llm_adjustment_available": True,
            "reasoning": prediction_result.get("reasoning", ""),
            "evidence": prediction_result.get("evidence_summary", []),
            "risk_notes": prediction_result.get("risk_notes", ""),
            "source": "llm_agent",
        },
        "warnings": warnings,
    }


def _extract_emitted_prediction(output_text: str) -> dict[str, Any] | None:
    """Extract predictions from LLM output in various formats.

    Handles:
    - JSON tool call results: {"predictions": [...], "reasoning": "..."}
    - Markdown tables with price columns
    - JSON code blocks containing predictions
    - Plain text float lists: [32.89, 32.95, ...]
    """
    if not output_text:
        return None

    import re

    # --- Strategy 1: Direct JSON parse ---
    try:
        parsed = json.loads(output_text)
        if isinstance(parsed, dict) and "predictions" in parsed:
            return _validate_extracted(parsed)
    except Exception:
        pass

    # --- Strategy 2: JSON objects containing "predictions" key ---
    candidates = re.findall(r'\{[^{}]*"predictions"\s*:\s*\[[^\]]*\][^{}]*\}', output_text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "predictions" in parsed:
                return _validate_extracted(parsed)
        except Exception:
            continue

    # --- Strategy 3: Markdown table extraction ---
    table_predictions = _extract_from_markdown_table(output_text)
    if table_predictions and len(table_predictions) >= 2:
        reasoning = _extract_reasoning_section(output_text)
        evidence = _extract_evidence_items(output_text)
        risks = _extract_risk_section(output_text)
        return {
            "accepted": True,
            "predictions": table_predictions,
            "reasoning": reasoning or "Extracted from markdown output.",
            "evidence_summary": evidence if evidence else ["Evidence extracted from LLM analysis."],
            "risk_notes": risks or "Auto-extracted from LLM output.",
        }

    # --- Strategy 4: JSON code blocks ---
    json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', output_text, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict) and "predictions" in parsed:
                return _validate_extracted(parsed)
        except Exception:
            continue

    # --- Strategy 5: Float lists in brackets ---
    float_lists = re.findall(r'\[(\s*-?\d+\.?\d*\s*(?:,\s*-?\d+\.?\d*\s*)*)\]', output_text)
    if float_lists:
        # Score each list: prefer lists with length matching typical horizon (3-10)
        for match in float_lists:
            try:
                values = [float(x.strip()) for x in match.split(",")]
                if 3 <= len(values) <= 15:
                    reasoning = _extract_reasoning_section(output_text)
                    return {
                        "accepted": True,
                        "predictions": values,
                        "reasoning": reasoning or "Extracted from LLM output text.",
                        "evidence_summary": [],
                        "risk_notes": "Auto-extracted; reasoning partially captured.",
                    }
            except Exception:
                continue

    # --- Strategy 6: Dollar amounts in text ---
    dollar_prices = _extract_dollar_prices(output_text)
    if dollar_prices and len(dollar_prices) >= 2:
        reasoning = _extract_reasoning_section(output_text)
        return {
            "accepted": True,
            "predictions": dollar_prices,
            "reasoning": reasoning or "Extracted from dollar amounts in LLM output.",
            "evidence_summary": [],
            "risk_notes": "Auto-extracted from dollar amounts.",
        }

    return None


def _validate_extracted(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and normalize a parsed prediction dict."""
    predictions = parsed.get("predictions")
    if not isinstance(predictions, list) or len(predictions) == 0:
        return None
    try:
        preds = [float(v) for v in predictions]
    except (ValueError, TypeError):
        return None
    return {
        "accepted": True,
        "predictions": preds,
        "reasoning": str(parsed.get("reasoning", "")),
        "evidence_summary": (
            parsed.get("evidence_summary", [])
            if isinstance(parsed.get("evidence_summary"), list)
            else [str(parsed.get("evidence_summary", ""))]
        ),
        "risk_notes": str(parsed.get("risk_notes", "")),
    }


def _extract_from_markdown_table(text: str) -> list[float] | None:
    """Extract predicted prices from a markdown table.

    Looks for tables like:
    | Date | Predicted Close |
    |------|----------------|
    | 2016-03-18 | $32.89 |
    """
    import re
    prices: list[float] = []

    # Find dollar amounts in table rows
    table_rows = re.findall(
        r'\|\s*[^|]*\|\s*\$?(\d+\.?\d*)\s*\|', text
    )

    for row in table_rows:
        try:
            val = float(row)
            if val > 0:
                prices.append(val)
        except ValueError:
            continue

    if prices:
        return prices

    # Alternative: find "Predicted Close" or "Predicted" column
    lines = text.split("\n")
    price_col_idx = None
    for line in lines:
        if "|" in line and ("predicted" in line.lower() or "close" in line.lower() or "price" in line.lower()):
            cells = [c.strip().lower() for c in line.split("|")]
            for i, cell in enumerate(cells):
                if cell in ("predicted close", "predicted", "price", "close", "forecast"):
                    price_col_idx = i
                    break
            if price_col_idx is not None:
                break

    if price_col_idx is not None:
        for line in lines:
            if "|" in line and not line.strip().startswith("|--"):
                cells = [c.strip() for c in line.split("|")]
                if len(cells) > price_col_idx:
                    val_str = cells[price_col_idx].replace("$", "").strip()
                    try:
                        val = float(val_str)
                        if val > 0:
                            prices.append(val)
                    except ValueError:
                        continue

    return prices if prices else None


def _extract_dollar_prices(text: str) -> list[float] | None:
    """Extract dollar amounts that look like stock prices."""
    import re
    dollar_pattern = re.findall(r'\$(\d+\.\d{2})\b', text)
    if not dollar_pattern:
        return None
    prices = []
    for p in dollar_pattern:
        try:
            val = float(p)
            if 0.5 < val < 10000:
                prices.append(val)
        except ValueError:
            continue
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in prices:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique if len(unique) >= 2 else None


def _extract_reasoning_section(text: str) -> str | None:
    """Extract reasoning/analysis from LLM output."""
    import re
    patterns = [
        r'#+\s*(?:Decision\s*)?Rationale\s*\n+(.*?)(?=\n#+\s|\n---|\Z)',
        r'#+\s*Analysis\s*\n+(.*?)(?=\n#+\s|\n---|\Z)',
        r'Reasoning:\s*\n+(.*?)(?=\n\s*(?:Evidence|Risk|Prediction|Constraints):|\Z)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:2000]
    # Fallback: first substantial paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    if paragraphs:
        return paragraphs[0][:2000]
    return None


def _extract_evidence_items(text: str) -> list[str]:
    """Extract evidence bullet points from LLM output."""
    import re
    evidence: list[str] = []

    # Look for evidence section
    ev_section = re.search(
        r'(?:Evidence|Bullish signals?|Supporting evidence)[:\s]*(.*?)(?=\n(?:Tempering|Bearish|Risk|Constraints|$))',
        text, re.IGNORECASE | re.DOTALL
    )
    if ev_section:
        section_text = ev_section.group(1)
    else:
        section_text = text

    # Extract numbered or bulleted items
    bullets = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*\*?|\-)\s*(.*?)(?=\n\s*(?:\d+\.|\*\*?|\-|$))', section_text)
    for bullet in bullets:
        clean = bullet.strip()[:300]
        if clean and len(clean) > 10:
            evidence.append(clean)
    return evidence[:6]


def _extract_risk_section(text: str) -> str | None:
    """Extract risk notes from LLM output."""
    import re
    patterns = [
        r'#+\s*Risk(?:s| Notes)?\s*\n+(.*?)(?=\n#+\s|\n---|\Z)',
        r'Tempering factors?[:\s]*(.*?)(?=\n(?:#|My final|Conclusion|\Z))',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:1000]
    return None


__all__ = [
    "STRATEGIST_AGENT_SYSTEM_PROMPT",
    "create_strategist_agent",
    "run_strategist",
]
