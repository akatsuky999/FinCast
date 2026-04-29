from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

from fincast.tools.dataloader import DEFAULT_MANIFEST_PATH, gather_forecast_inputs as deterministic_gather_forecast_inputs


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PACKAGE_ROOT / ".env"


INVESTIGATOR_AGENT_PROMPT = dedent(
    """
    You are InvestigatorAgent for FinCast, a financial time-series forecasting workflow.

    For every request:
      1. Call `gather_forecast_inputs` exactly once with the dataset_name, window_offset, and forecast_horizon provided by the user.
      2. Read the returned deterministic packet carefully.
      3. Return a compact JSON object with exactly these keys:
         - news_summary: brief summary of the recent aligned news context.
         - financial_state_summary: brief summary of price trend, volatility, drawdown, volume/news activity, and regime.
         - risk_notes: concise warnings about instability, sparse news, high volatility, or weak evidence.

    Strict constraints:
      - Do not forecast prices.
      - Do not output prediction values.
      - Do not use or infer future target values.
      - Do not cite news outside the look-back window.
      - If the packet has no meaningful news, say so plainly.
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


def _parse_summary(text: str) -> dict[str, Any]:
    raw_text = text.strip()
    if "```" in raw_text:
        parts = raw_text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                raw_text = candidate
                break
    if not raw_text.startswith("{"):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if 0 <= start < end:
            raw_text = raw_text[start : end + 1]
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {
            "llm_summary_available": True,
            "news_summary": text.strip(),
            "financial_state_summary": "",
            "risk_notes": "",
        }
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "llm_summary_available": True,
        "news_summary": str(parsed.get("news_summary", "")).strip(),
        "financial_state_summary": str(parsed.get("financial_state_summary", "")).strip(),
        "risk_notes": str(parsed.get("risk_notes", "")).strip(),
    }


def build_investigator_agent(model_name: str | None = None, manifest_path: str | Path = DEFAULT_MANIFEST_PATH):
    """Build the pydantic-ai Investigator Agent.

    The import is intentionally lazy so deterministic tools work even when
    pydantic-ai is not installed in the active environment.
    """

    try:
        from pydantic_ai import Agent, RunContext  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "pydantic_ai is required for the LLM Investigator Agent. "
            "The deterministic dataloader can still be used without it."
        ) from exc
    globals()["RunContext"] = RunContext

    resolved_model = model_name or _model_name_from_env()
    if not resolved_model:
        raise RuntimeError("No pydantic-ai model configured. Set PYA_MODEL or MODEL in FinCast/.env.")

    agent = Agent(resolved_model, instructions=INVESTIGATOR_AGENT_PROMPT)

    @agent.tool
    def gather_forecast_inputs(
        ctx: RunContext[None],
        dataset_name: str,
        window_offset: int = 0,
        forecast_horizon: int | None = None,
    ) -> dict[str, Any]:
        return deterministic_gather_forecast_inputs(
            dataset_name=dataset_name,
            window_offset=window_offset,
            forecast_horizon=forecast_horizon,
            manifest_path=manifest_path,
        )

    return agent


def run_investigator(
    dataset_name: str,
    window_offset: int = 0,
    forecast_horizon: int | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    use_llm: bool = True,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return a FinCast Investigator packet with optional LLM summaries."""

    packet = deterministic_gather_forecast_inputs(
        dataset_name=dataset_name,
        window_offset=window_offset,
        forecast_horizon=forecast_horizon,
        manifest_path=manifest_path,
    )
    if not use_llm:
        return packet

    try:
        agent = build_investigator_agent(model_name=model_name, manifest_path=manifest_path)
        prompt = json.dumps(
            {
                "dataset_name": dataset_name,
                "window_offset": int(window_offset),
                "forecast_horizon": forecast_horizon,
            }
        )
        result = agent.run_sync(prompt)
        packet["llm_summary"] = _parse_summary(_result_output(result))
    except Exception as exc:
        packet.setdefault("warnings", []).append(f"LLM Investigator summary unavailable: {exc}")
        packet["llm_summary"] = {
            "llm_summary_available": False,
            "news_summary": "",
            "financial_state_summary": "",
            "risk_notes": "",
        }
    return packet


def main() -> None:
    dataset_name = os.getenv("FINCAST_DATASET", "FinCastPrice_NVDA")
    window_offset = int(os.getenv("FINCAST_WINDOW_OFFSET", "0"))
    forecast_horizon = int(os.getenv("FINCAST_FORECAST_HORIZON", "5"))
    use_llm = os.getenv("FINCAST_USE_LLM", "1") != "0"
    packet = run_investigator(
        dataset_name=dataset_name,
        window_offset=window_offset,
        forecast_horizon=forecast_horizon,
        use_llm=use_llm,
    )
    print(json.dumps(packet, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "INVESTIGATOR_AGENT_PROMPT",
    "build_investigator_agent",
    "run_investigator",
]
