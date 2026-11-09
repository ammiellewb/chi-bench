"""Provider routing and runner-control tests for the openai-agents harness."""

from __future__ import annotations

from typing import Any

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from chi_bench.experiment.agents.openai_agents_harness import (
    OpenAIAgentsHarness,
    _build_atif_trajectory,
)


INKLING_MODEL = "thinkingmachines/Inkling:peft:262144"
NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:peft:262144"


def test_empty_trace_with_error_has_sequential_fallback_steps() -> None:
    trajectory = _build_atif_trajectory(
        records=[],
        model_name=INKLING_MODEL,
        final_output=None,
        error="ModelBehaviorError: Tool bash not found in agent chi_bench-agent",
        agent_version="0.13.6",
        cost_usd=None,
    )

    payload = trajectory.to_json_dict()
    assert [step["step_id"] for step in payload["steps"]] == [1, 2]
    assert [step["message"] for step in payload["steps"]] == [
        "No openai-agents trace captured.",
        "[run error] ModelBehaviorError: Tool bash not found in agent chi_bench-agent",
    ]


def test_thinkingmachines_model_routes_to_tinker_chat_completions() -> None:
    env = OpenAIAgentsHarness._resolve_routing(
        INKLING_MODEL,
        {"TINKER_API_KEY": "test-tinker-key"},
    )

    assert env == {
        "OPENAI_API_KEY": "test-tinker-key",
        "OPENAI_BASE_URL": "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        "OPENAI_AGENTS_MODEL": INKLING_MODEL,
        "OPENAI_AGENTS_API_MODE": "chat_completions",
    }


def test_thinkingmachines_model_requires_tinker_api_key() -> None:
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        OpenAIAgentsHarness._resolve_routing(INKLING_MODEL, {})


def test_thinkingmachines_route_ignores_generic_openai_base_url() -> None:
    env = OpenAIAgentsHarness._resolve_routing(
        INKLING_MODEL,
        {
            "TINKER_API_KEY": "test-tinker-key",
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
        },
    )

    assert env["OPENAI_API_KEY"] == "test-tinker-key"
    assert env["OPENAI_BASE_URL"] == OpenAIAgentsHarness.TINKER_BASE_URL
    assert env["OPENAI_AGENTS_API_MODE"] == "chat_completions"


def test_existing_openai_and_openrouter_routes_are_unchanged() -> None:
    assert OpenAIAgentsHarness._resolve_routing(
        "openai/gpt-5.6-sol",
        {"OPENAI_API_KEY": "test-openai-key"},
    ) == {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_BASE_URL": OpenAIAgentsHarness.OPENAI_DIRECT_BASE_URL,
        "OPENAI_AGENTS_MODEL": "gpt-5.6-sol",
    }
    assert OpenAIAgentsHarness._resolve_routing(
        "moonshotai/kimi-k3",
        {"OPENROUTER_API_KEY": "test-openrouter-key"},
    ) == {
        "OPENAI_API_KEY": "test-openrouter-key",
        "OPENAI_BASE_URL": OpenAIAgentsHarness.OPENROUTER_BASE_URL,
        "OPENAI_AGENTS_MODEL": "moonshotai/kimi-k3",
    }


def test_explicit_tinker_route_uses_tinker_for_non_thinkingmachines_model() -> None:
    env = OpenAIAgentsHarness._resolve_routing(
        NEMOTRON_MODEL,
        {
            "TINKER_API_KEY": "test-tinker-key",
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
        },
        provider_route="tinker",
    )

    assert env == {
        "OPENAI_API_KEY": "test-tinker-key",
        "OPENAI_BASE_URL": OpenAIAgentsHarness.TINKER_BASE_URL,
        "OPENAI_AGENTS_MODEL": NEMOTRON_MODEL,
        "OPENAI_AGENTS_API_MODE": "chat_completions",
    }


def test_explicit_tinker_route_requires_tinker_api_key() -> None:
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        OpenAIAgentsHarness._resolve_routing(
            NEMOTRON_MODEL,
            {"OPENROUTER_API_KEY": "test-openrouter-key"},
            provider_route="tinker",
        )


def test_nemotron_without_explicit_route_still_uses_openrouter() -> None:
    env = OpenAIAgentsHarness._resolve_routing(
        NEMOTRON_MODEL,
        {
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "TINKER_API_KEY": "test-tinker-key",
        },
    )

    assert env == {
        "OPENAI_API_KEY": "test-openrouter-key",
        "OPENAI_BASE_URL": OpenAIAgentsHarness.OPENROUTER_BASE_URL,
        "OPENAI_AGENTS_MODEL": NEMOTRON_MODEL,
    }


def test_provider_route_flag_is_exposed_to_cli_and_env() -> None:
    flags = {flag.kwarg: flag for flag in OpenAIAgentsHarness.CLI_FLAGS}

    assert "provider_route" in flags
    flag = flags["provider_route"]
    assert flag.cli == "--provider-route"
    assert flag.type == "enum"
    assert flag.choices == ["auto", "tinker", "litellm"]
    assert flag.env_fallback == "OPENAI_AGENTS_PROVIDER_ROUTE"


@pytest.mark.asyncio
async def test_install_pins_compatible_openai_sdk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = OpenAIAgentsHarness(logs_dir=tmp_path)
    captured: dict[str, Any] = {}

    async def fake_exec_as_root(environment, *, command: str):
        captured.update(environment=environment, command=command)

    monkeypatch.setattr(harness, "exec_as_root", fake_exec_as_root)
    environment = object()

    await harness.install(environment)

    assert captured == {
        "environment": environment,
        "command": (
            "uv pip install --no-cache-dir --python /workspace/.venv "
            "openai-agents==0.13.6 openai==2.36.0"
        ),
    }


@pytest.mark.asyncio
async def test_api_mode_and_reasoning_effort_are_forwarded_to_runner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.6-sol",
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
        api_mode="responses",
        reasoning_effort="high",
    )
    captured: dict[str, Any] = {}

    async def fake_exec_as_agent(environment, *, command: str, env: dict[str, str]):
        captured.update(environment=environment, command=command, env=env)

    monkeypatch.setattr(harness, "exec_as_agent", fake_exec_as_agent)

    environment = object()
    await harness.run("Complete the task", environment, AgentContext())

    assert captured["environment"] is environment
    assert captured["env"]["OPENAI_AGENTS_API_MODE"] == "responses"
    assert captured["env"]["OPENAI_AGENTS_REASONING_EFFORT"] == "high"


@pytest.mark.asyncio
async def test_provider_route_is_forwarded_to_routing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-tinker-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name=NEMOTRON_MODEL,
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
        provider_route="tinker",
        api_mode="chat_completions",
    )
    captured: dict[str, Any] = {}

    async def fake_exec_as_agent(environment, *, command: str, env: dict[str, str]):
        captured.update(environment=environment, command=command, env=env)

    monkeypatch.setattr(harness, "exec_as_agent", fake_exec_as_agent)

    await harness.run("Complete the task", object(), AgentContext())

    assert captured["env"]["OPENAI_API_KEY"] == "test-tinker-key"
    assert captured["env"]["OPENAI_BASE_URL"] == OpenAIAgentsHarness.TINKER_BASE_URL
    assert captured["env"]["OPENAI_AGENTS_MODEL"] == NEMOTRON_MODEL
    assert captured["env"]["OPENAI_AGENTS_API_MODE"] == "chat_completions"


@pytest.mark.asyncio
async def test_tinker_route_forces_chat_completions_over_api_mode_flag(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-tinker-key")
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name=NEMOTRON_MODEL,
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
        provider_route="tinker",
        api_mode="responses",
    )
    captured: dict[str, Any] = {}

    async def fake_exec_as_agent(environment, *, command: str, env: dict[str, str]):
        captured.update(environment=environment, command=command, env=env)

    monkeypatch.setattr(harness, "exec_as_agent", fake_exec_as_agent)

    await harness.run("Complete the task", object(), AgentContext())

    assert captured["env"]["OPENAI_AGENTS_API_MODE"] == "chat_completions"


@pytest.mark.asyncio
async def test_tinker_route_ignores_api_mode_from_extra_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-tinker-key")
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name=NEMOTRON_MODEL,
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
        provider_route="tinker",
    )
    saved_extra_env = {"OPENAI_AGENTS_API_MODE": "responses"}
    harness._extra_env = saved_extra_env
    captured: dict[str, Any] = {}

    async def fake_exec_as_agent(environment, *, command: str, env: dict[str, str]):
        captured["env"] = {**env, **harness._extra_env}

    monkeypatch.setattr(harness, "exec_as_agent", fake_exec_as_agent)

    await harness.run("Complete the task", object(), AgentContext())

    assert captured["env"]["OPENAI_AGENTS_API_MODE"] == "chat_completions"
    assert harness._extra_env is saved_extra_env
    assert saved_extra_env == {"OPENAI_AGENTS_API_MODE": "responses"}


@pytest.mark.asyncio
async def test_non_tinker_route_keeps_api_mode_from_extra_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.6-sol",
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
        api_mode="chat_completions",
    )
    saved_extra_env = {"OPENAI_AGENTS_API_MODE": "responses"}
    harness._extra_env = saved_extra_env
    captured: dict[str, Any] = {}

    async def fake_exec_as_agent(environment, *, command: str, env: dict[str, str]):
        captured["env"] = {**env, **harness._extra_env}

    monkeypatch.setattr(harness, "exec_as_agent", fake_exec_as_agent)

    await harness.run("Complete the task", object(), AgentContext())

    assert captured["env"]["OPENAI_AGENTS_API_MODE"] == "responses"
    assert harness._extra_env is saved_extra_env
    assert saved_extra_env == {"OPENAI_AGENTS_API_MODE": "responses"}


@pytest.mark.asyncio
async def test_routing_failure_restores_original_extra_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "host-openai-key")
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name=NEMOTRON_MODEL,
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
        provider_route="tinker",
    )
    saved_extra_env = {
        "OPENAI_API_KEY": "extra-openai-key",
        "OPENAI_BASE_URL": "https://example.invalid/v1",
        "KEEP_ME": "test-value",
    }
    harness._extra_env = saved_extra_env

    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        await harness.run("Complete the task", object(), AgentContext())

    assert harness._extra_env is saved_extra_env
    assert harness._extra_env == {
        "OPENAI_API_KEY": "extra-openai-key",
        "OPENAI_BASE_URL": "https://example.invalid/v1",
        "KEEP_ME": "test-value",
    }


# ── LiteLLM route + mechanisms forwarding ─────────────────────────────


def test_litellm_prefix_routes_to_proxy_and_strips_prefix() -> None:
    env = OpenAIAgentsHarness._resolve_routing(
        "litellm/qwen3-35b",
        {
            "LITELLM_BASE_URL": "http://gpubox:4000/v1",
            "LITELLM_API_KEY": "sk-litellm",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
            "OPENROUTER_API_KEY": "sk-or",
        },
    )
    assert env == {
        "OPENAI_API_KEY": "sk-litellm",
        "OPENAI_BASE_URL": "http://gpubox:4000/v1",
        "OPENAI_AGENTS_MODEL": "qwen3-35b",
        "OPENAI_AGENTS_API_MODE": "chat_completions",
    }


def test_explicit_litellm_route_keeps_model_id_and_defaults_key() -> None:
    env = OpenAIAgentsHarness._resolve_routing(
        "google/gemma-4-12b",
        {"LITELLM_BASE_URL": "http://gpubox:8000/v1"},
        provider_route="litellm",
    )
    assert env["OPENAI_AGENTS_MODEL"] == "google/gemma-4-12b"
    assert env["OPENAI_API_KEY"] == "litellm"
    assert env["OPENAI_BASE_URL"] == "http://gpubox:8000/v1"


def test_litellm_route_requires_base_url() -> None:
    with pytest.raises(RuntimeError, match="LITELLM_BASE_URL"):
        OpenAIAgentsHarness._resolve_routing("litellm/qwen3-35b", {})


def test_unknown_mechanism_fails_fast(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown mechanism"):
        OpenAIAgentsHarness(logs_dir=tmp_path, model_name="openai/gpt-5.5", mechanisms="prefight")


@pytest.mark.asyncio
async def test_mechanisms_bundle_is_uploaded_and_flags_forwarded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://gpubox:4000/v1")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name="litellm/qwen3-35b",
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
        mechanisms="preflight,consent_gate",
        progress_nudge_turn=8,
        turn_budget_warning=5,
        api_mode="responses",
    )
    uploads: list[tuple[list[str], str]] = []
    commands: list[tuple[str, str]] = []
    captured: dict[str, Any] = {}

    class FakeEnvironment:
        async def upload_dir(self, source_dir, target_dir: str) -> None:
            files = sorted(
                str(p.relative_to(source_dir)) for p in source_dir.rglob("*") if p.is_file()
            )
            uploads.append((files, target_dir))

    async def fake_exec_as_root(environment, *, command: str, timeout_sec=None, **_kw):
        commands.append(("root", command))

    async def fake_exec_as_agent(environment, *, command: str, env: dict[str, str]):
        captured.update(command=command, env=env)

    monkeypatch.setattr(harness, "exec_as_root", fake_exec_as_root)
    monkeypatch.setattr(harness, "exec_as_agent", fake_exec_as_agent)

    await harness.run("Complete the task", FakeEnvironment(), AgentContext())

    assert [m.name for m in harness.mechanisms] == ["preflight", "consent_gate"]
    files, target = uploads[0]
    assert target == "/logs/agent/mechanisms"
    assert {"bin/preflight_hook.py", "bin/consent_gate_hook.py", "bin/observe_hook.py"} <= set(
        files
    )
    assert "bin/policy_trace_hook.py" not in files
    assert commands == [("root", "chmod -R a+rwX /logs/agent/mechanisms")]
    env = captured["env"]
    assert env["OPENAI_AGENTS_MECHANISMS"] == "preflight,consent_gate"
    assert env["CHI_BENCH_MECH_STATE_DIR"] == "/logs/agent/mechanisms"
    assert env["CHI_BENCH_MECH_HANDBOOK_ROOT"].endswith("managed-care-operations-handbook")
    assert env["OPENAI_AGENTS_PROGRESS_NUDGE_TURN"] == "8"
    assert env["OPENAI_AGENTS_TURN_BUDGET_WARNING"] == "5"
    assert env["OPENAI_AGENTS_MODEL"] == "qwen3-35b"
    assert env["OPENAI_BASE_URL"] == "http://gpubox:4000/v1"
    # An explicit api_mode still wins over the LiteLLM chat-completions default.
    assert env["OPENAI_AGENTS_API_MODE"] == "responses"


@pytest.mark.asyncio
async def test_baseline_run_forwards_no_mechanism_or_nudge_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    for var in ("OPENAI_AGENTS_MECHANISMS", "OPENAI_AGENTS_PROGRESS_NUDGE_TURN"):
        monkeypatch.delenv(var, raising=False)
    harness = OpenAIAgentsHarness(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        mcp_servers=[
            MCPServerConfig(
                name="chi_bench",
                transport="streamable-http",
                url="http://chi-bench-server:8000/mcp",
            )
        ],
    )
    captured: dict[str, Any] = {}

    async def fake_exec_as_agent(environment, *, command: str, env: dict[str, str]):
        captured.update(env=env)

    monkeypatch.setattr(harness, "exec_as_agent", fake_exec_as_agent)
    await harness.run("Complete the task", object(), AgentContext())

    assert harness.mechanisms == []
    assert not any(
        k.startswith(("OPENAI_AGENTS_MECH", "OPENAI_AGENTS_PROGRESS", "OPENAI_AGENTS_TURN_B"))
        for k in captured["env"]
    )
