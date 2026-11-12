"""Build the workspace-judge runner selected by ``CHI_BENCH_JUDGE_PROVIDER``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chi_bench.llm_routes import judge_route


def make_judge_runner(
    *,
    timeout_seconds: int,
    max_retries: int = 1,
    debug_dir: Path | None = None,
) -> Any:
    """Claude Code CLI runner by default; OpenAI-compatible runner for ``openai``."""
    route = judge_route()
    if route.is_openai:
        from chi_bench.verifier.judge.openai_runner import OpenAICompatJudgeRunner

        return OpenAICompatJudgeRunner(
            route,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            debug_dir=debug_dir,
        )
    from chi_bench.verifier.judge.claude_runner import ClaudeAgentRunner

    return ClaudeAgentRunner(
        model=route.model or "claude-opus-4-7",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        debug_dir=debug_dir,
    )


def judge_provenance() -> dict[str, Any]:
    """``judge_model`` plus the route it was reached through (no credentials)."""
    route = judge_route()
    return {
        "judge_model": route.model,
        "judge_provider": route.provider,
        "judge_base_url": route.base_url if route.is_openai else None,
    }
