__all__ = [
    "BASELINE_AGENT_PROMPT",
    "STRATEGIST_AGENT_SYSTEM_PROMPT",
    "BRIEFING_AGENT_PROMPT",
    "REFLECTOR_AGENT_PROMPT",
    "attach_llm_explanation",
    "create_strategist_agent",
    "build_baseline_agent",
    "build_baseline_packet",
    "build_briefing_agent",
    "build_reflector_agent",
    "run_fincast_pipeline",
    "run_strategist",
    "run_briefing",
    "reflect_forecast",
]


def __getattr__(name):
    if name in {
        "BRIEFING_AGENT_PROMPT",
        "build_briefing_agent",
        "run_briefing",
    }:
        from . import briefing_agent

        return getattr(briefing_agent, name)
    if name in {
        "BASELINE_AGENT_PROMPT",
        "attach_llm_explanation",
        "build_baseline_agent",
        "build_baseline_packet",
    }:
        from . import baseline_agent

        return getattr(baseline_agent, name)
    if name in {
        "STRATEGIST_AGENT_SYSTEM_PROMPT",
        "create_strategist_agent",
        "run_strategist",
    }:
        from . import strategist_agent

        return getattr(strategist_agent, name)
    if name in {
        "REFLECTOR_AGENT_PROMPT",
        "build_reflector_agent",
        "reflect_forecast",
    }:
        from . import reflector_agent

        return getattr(reflector_agent, name)
    if name in {"run_fincast_pipeline"}:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(name)
