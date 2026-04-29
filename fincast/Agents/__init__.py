__all__ = [
    "BASELINE_AGENT_PROMPT",
    "GENERATOR_AGENT_PROMPT",
    "INVESTIGATOR_AGENT_PROMPT",
    "REFLECTOR_AGENT_PROMPT",
    "attach_llm_explanation",
    "build_generator_agent",
    "build_baseline_agent",
    "build_baseline_packet",
    "build_investigator_agent",
    "build_reflector_agent",
    "run_fincast_pipeline",
    "run_generator",
    "run_investigator",
    "reflect_forecast",
]


def __getattr__(name):
    if name in {
        "INVESTIGATOR_AGENT_PROMPT",
        "build_investigator_agent",
        "run_investigator",
    }:
        from . import investigator_agent

        return getattr(investigator_agent, name)
    if name in {
        "BASELINE_AGENT_PROMPT",
        "attach_llm_explanation",
        "build_baseline_agent",
        "build_baseline_packet",
    }:
        from . import baseline_agent

        return getattr(baseline_agent, name)
    if name in {
        "GENERATOR_AGENT_PROMPT",
        "build_generator_agent",
        "run_generator",
    }:
        from . import generator_agent

        return getattr(generator_agent, name)
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
