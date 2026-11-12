"""Composition tests for the `claude-code-mechanisms` harness."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import pytest

from chi_bench.experiment.agents.claude_code_mechanisms_harness import (
    ClaudeCodeMechanismsHarness,
)
from chi_bench.experiment.agents.mechanisms import (
    MECHANISMS,
    build_settings,
    compose_prompt,
    resolve_mechanisms,
    write_bundle,
)


def test_resolve_mechanisms_parses_lists_and_all() -> None:
    assert resolve_mechanisms(None) == []
    assert resolve_mechanisms("") == []
    assert [m.name for m in resolve_mechanisms("policy_trace, preflight")] == [
        "preflight",
        "policy_trace",
    ]
    assert [m.name for m in resolve_mechanisms("all")] == list(MECHANISMS)
    with pytest.raises(ValueError, match="unknown mechanism"):
        resolve_mechanisms("preflight,typo")


def test_each_mechanism_is_an_independent_increment() -> None:
    """Enabling one more mechanism adds exactly its own hooks and prompt."""
    base = resolve_mechanisms("preflight")
    plus = resolve_mechanisms("preflight,criteria_ledger")
    base_settings = build_settings(base)
    plus_settings = build_settings(plus)
    base_pre = base_settings["hooks"]["PreToolUse"]
    plus_pre = plus_settings["hooks"]["PreToolUse"]
    assert len(plus_pre) == len(base_pre) + 1
    assert plus_pre[:1] == base_pre[:1]
    assert "criteria_ledger_hook.py" in plus_pre[-1]["hooks"][0]["command"]
    assert compose_prompt(plus).startswith(compose_prompt(base))


def test_build_settings_adds_observer_only_when_mechanisms_active() -> None:
    assert build_settings([]) == {}
    settings = build_settings(resolve_mechanisms("consent_gate"))
    post_commands = [h["hooks"][0]["command"] for h in settings["hooks"]["PostToolUse"]]
    assert any("observe_hook.py" in c for c in post_commands)
    assert any("consent_gate_hook.py" in c for c in post_commands)
    assert all(c.startswith("python3 -I /logs/agent/mechanisms/bin/") for c in post_commands)


def test_write_bundle_materializes_scripts_skills_and_manifest(tmp_path: Path) -> None:
    dest = write_bundle(
        resolve_mechanisms("policy_trace"), tmp_path / "bundle", state_dir="/s", handbook_root="/h"
    )
    scripts = sorted(p.name for p in (dest / "bin").iterdir())
    assert scripts == [
        "mech_common.py",
        "observe_hook.py",
        "policy_locate.py",
        "policy_trace_hook.py",
    ]
    skill = (dest / "skills" / "policy-locator" / "SKILL.md").read_text()
    assert "python3 -I /s/bin/policy_locate.py" in skill and "/h" in skill
    settings = json.loads((dest / "settings.json").read_text())
    assert (
        settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        == "python3 -I /s/bin/policy_trace_hook.py"
    )
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["mechanisms"] == ["policy_trace"] and manifest["skills"] == [
        "policy-locator/SKILL.md"
    ]


def test_harness_without_mechanisms_is_the_stock_baseline(tmp_path: Path) -> None:
    harness = ClaudeCodeMechanismsHarness(
        logs_dir=tmp_path, model_name="anthropic/claude-sonnet-5-5"
    )
    assert harness.mechanisms == []
    assert "--append-system-prompt" not in harness.build_cli_flags()
    assert harness._build_register_skills_command() is None
    assert ClaudeCodeMechanismsHarness.name() == "claude-code-mechanisms"


def test_harness_appends_quoted_prompt_and_merges_user_prompt(tmp_path: Path) -> None:
    harness = ClaudeCodeMechanismsHarness(
        logs_dir=tmp_path,
        mechanisms="preflight,policy_trace",
        append_system_prompt="Always answer in English.",
        model_name="anthropic/claude-sonnet-5-5",
    )
    flags = harness.build_cli_flags()
    tokens = shlex.split(flags)
    prompt = tokens[tokens.index("--append-system-prompt") + 1]
    assert prompt.startswith("Always answer in English.")
    assert "### Mechanism: preflight" in prompt and "### Mechanism: traceable policy use" in prompt
    assert "### Mechanism: assess criteria" not in prompt
    assert "--permission-mode=bypassPermissions" in flags


def test_harness_exports_state_env_and_skill_copy(tmp_path: Path) -> None:
    harness = ClaudeCodeMechanismsHarness(
        logs_dir=tmp_path,
        mechanisms="policy_trace",
        mechanisms_state_dir="/logs/agent/mech-x",
        model_name="anthropic/claude-sonnet-5-5",
    )
    env = harness.resolve_env_vars()
    assert env["CHI_BENCH_MECH_STATE_DIR"] == "/logs/agent/mech-x"
    assert env["CHI_BENCH_MECH_HANDBOOK_ROOT"].endswith("managed-care-operations-handbook")
    assert "/logs/agent/mech-x/skills" in (harness._build_register_skills_command() or "")
    assert "/logs/agent/mech-x" in harness.composed_system_prompt()


def test_harness_preserves_user_skills_dir(tmp_path: Path) -> None:
    harness = ClaudeCodeMechanismsHarness(
        logs_dir=tmp_path, mechanisms="policy_trace", skills_dir="/opt/user-skills"
    )
    command = harness._build_register_skills_command() or ""
    assert "/opt/user-skills" in command and "/logs/agent/mechanisms/skills" in command


def test_litellm_route_points_claude_code_at_the_proxy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    harness = ClaudeCodeMechanismsHarness(
        logs_dir=tmp_path,
        mechanisms="preflight",
        model_name="litellm/qwen3-35b",
        extra_env={
            "LITELLM_BASE_URL": "http://gpubox:4000",
            "LITELLM_API_KEY": "proxy-key",
            "ANTHROPIC_API_KEY": "native-key",
        },
    )
    assert harness.model_name == "qwen3-35b"
    assert harness._parsed_model_name == "qwen3-35b" and harness._parsed_model_provider is None
    assert harness.extra_env["ANTHROPIC_API_KEY"] == "proxy-key"
    assert harness.extra_env["ANTHROPIC_BASE_URL"] == "http://gpubox:4000"
    assert os.environ["ANTHROPIC_BASE_URL"] == "http://gpubox:4000"
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


def test_litellm_route_requires_base_url_and_rejects_unknown_routes(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="LITELLM_BASE_URL"):
        ClaudeCodeMechanismsHarness(logs_dir=tmp_path, provider_route="litellm", model_name="x")
    with pytest.raises(ValueError, match="provider_route"):
        ClaudeCodeMechanismsHarness(logs_dir=tmp_path, provider_route="tinker", model_name="x")


def test_completion_mechanism_registers_a_stop_hook_without_matcher(tmp_path: Path) -> None:
    settings = build_settings(resolve_mechanisms("completion"))
    stop = settings["hooks"]["Stop"]
    assert len(stop) == 1 and "matcher" not in stop[0]
    assert stop[0]["hooks"][0]["command"].endswith("/bin/completion_hook.py")
    assert all("matcher" in e for e in settings["hooks"]["PostToolUse"])
    dest = write_bundle(resolve_mechanisms("completion"), tmp_path / "b")
    assert (dest / "bin" / "completion_hook.py").exists()
    assert "### Mechanism: finish the workflow before stopping" in (dest / "prompt.md").read_text()
    assert "completion" in [m.name for m in resolve_mechanisms("all")]
