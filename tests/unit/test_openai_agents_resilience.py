"""Tool-name resilience, stage timing, nudges, and in-process mechanism gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chi_bench.experiment.agents import openai_agents_runner
from chi_bench.experiment.agents.mechanisms import (
    MECHANISMS,
    resolve_mechanisms,
    write_bundle,
)
from chi_bench.experiment.agents.mechanisms.hooks import mech_common
from chi_bench.experiment.agents.openai_agents_resilience import (
    UNKNOWN_TOOL_SINK,
    MechanismGate,
    NudgePolicy,
    ResilientModel,
    ResilientModelProvider,
    StageLog,
    ToolCallRewrite,
    ToolGuard,
    ToolNameResolver,
    hook_tool_identity,
)

HOOKS_DIR = Path(mech_common.__file__).parent


def _fc(name: str, call_id: str, arguments: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=arguments)


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


class _InnerModel:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def get_response(self, system_instructions, input, model_settings, tools, *a, **kw):
        self.calls.append({"input": input, "tools": [t.name for t in tools]})
        return self.responses.pop(0)


def _response(*outputs: Any, usage: Any = None) -> SimpleNamespace:
    if usage is None:
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
        )
    return SimpleNamespace(output=list(outputs), usage=usage, response_id=None)


# ── resolver ───────────────────────────────────────────────────────────


def test_resolver_maps_separator_variants_and_mcp_prefix() -> None:
    resolver = ToolNameResolver(["local-fs-read", "chart__get_patient", "local-claim_done"])
    assert resolver.resolve("local-fs-read") == "local-fs-read"
    assert resolver.resolve("local_fs_read") == "local-fs-read"
    assert resolver.resolve("LocalFsRead") == "local-fs-read"
    assert resolver.resolve("chart.get_patient") == "chart__get_patient"
    assert resolver.resolve("mcp__chi_bench__chart.get_patient") == "chart__get_patient"
    assert resolver.resolve("local_claim_done") == "local-claim_done"
    assert resolver.resolve("made_up_tool") is None
    assert "local-fs-read" in resolver.suggestions("local_fs_reed")


def test_resolver_keeps_ambiguous_collapsed_names_exact_only() -> None:
    resolver = ToolNameResolver(["a-b", "a_b"])
    assert resolver.resolve("a.b") is None
    assert resolver.resolve("a-b") == "a-b"


# ── model wrapper ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_wrapper_resolves_aliases_and_routes_unknown_to_sink(tmp_path: Path) -> None:
    stages = StageLog(tmp_path / "stages.jsonl")
    rewrites: dict[str, ToolCallRewrite] = {}
    alias = _fc("local_fs_read", "c1", '{"path": "/x"}')
    unknown = _fc("read_file", "c2", '{"path": "/y"}')
    inner = _InnerModel([_response(alias, unknown)])
    model = ResilientModel(inner, stages=stages, rewrites=rewrites)
    tools = [_tool("local-fs-read"), _tool("local-bash"), _tool(UNKNOWN_TOOL_SINK)]

    await model.get_response(None, [{"role": "user", "content": "go"}], None, tools, None, [], None)

    assert inner.calls[0]["tools"] == ["local-fs-read", "local-bash"]
    assert alias.name == "local-fs-read"
    assert unknown.name == UNKNOWN_TOOL_SINK
    sink_args = json.loads(unknown.arguments)
    assert sink_args["name"] == "read_file" and sink_args["arguments"] == '{"path": "/y"}'
    assert (
        rewrites["c1"].resolved == "local-fs-read" and rewrites["c1"].requested == "local_fs_read"
    )
    assert rewrites["c2"].unresolved and rewrites["c2"].requested == "read_file"
    kinds = [
        json.loads(line)["kind"] for line in (tmp_path / "stages.jsonl").read_text().splitlines()
    ]
    assert kinds == ["tool_alias", "unknown_tool", "model_call"]
    model_call = stages.records[-1]
    assert model_call["input_tokens"] == 10 and model_call["cached_tokens"] == 2
    assert model_call["n_tool_calls"] == 2 and model_call["turn"] == 1


@pytest.mark.asyncio
async def test_model_wrapper_restores_sink_calls_in_history(tmp_path: Path) -> None:
    stages = StageLog(tmp_path / "stages.jsonl")
    rewrites = {"c2": ToolCallRewrite("read_file", None, '{"path": "/y"}')}
    inner = _InnerModel([_response()])
    model = ResilientModel(inner, stages=stages, rewrites=rewrites)
    history = [
        {"role": "user", "content": "go"},
        {"type": "function_call", "name": UNKNOWN_TOOL_SINK, "call_id": "c2", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "ERROR"},
    ]

    await model.get_response(None, history, None, [], None, [], None)

    sent = inner.calls[0]["input"]
    assert sent[1]["name"] == "read_file" and sent[1]["arguments"] == '{"path": "/y"}'
    assert history[1]["name"] == UNKNOWN_TOOL_SINK  # the SDK's own history is untouched


@pytest.mark.asyncio
async def test_turn_counting_ignores_retries_and_records_errors(tmp_path: Path) -> None:
    stages = StageLog(tmp_path / "stages.jsonl")

    class Flaky(_InnerModel):
        async def get_response(self, *args, **kwargs):
            if len(self.calls) == 0:
                self.calls.append({})
                raise RuntimeError("boom")
            return await super().get_response(*args, **kwargs)

    inner = Flaky([_response(), _response()])
    model = ResilientModel(inner, stages=stages, rewrites={})
    one = [{"role": "user", "content": "go"}]
    with pytest.raises(RuntimeError):
        await model.get_response(None, one, None, [], None, [], None)
    await model.get_response(None, one, None, [], None, [], None)  # retry: same turn
    await model.get_response(None, one + [{}, {}], None, [], None, [], None)

    turns = [r["turn"] for r in stages.records if r["kind"] == "model_call"]
    assert turns == [1, 1, 2]
    assert stages.records[0]["error"].startswith("RuntimeError")
    assert stages.partial_usage() == {"input_tokens": 20, "output_tokens": 10, "cached_tokens": 4}
    assert stages.summary()["model_errors"] == 1


def test_provider_wraps_once_per_name_and_honours_override() -> None:
    inner_provider = SimpleNamespace(get_model=lambda name: _InnerModel([]))
    wrapped: list[Any] = []

    def wrap(model: Any) -> ResilientModel:
        wrapped.append(model)
        return ResilientModel(model, stages=StageLog(Path("/nonexistent/x")), rewrites={})

    provider = ResilientModelProvider(inner_provider, wrap=wrap)
    first = provider.get_model("m")
    assert provider.get_model("m") is first and len(wrapped) == 1
    override = _InnerModel([])
    provider.override_model = override
    assert provider.get_model("other").inner is override


# ── nudges ─────────────────────────────────────────────────────────────


def test_nudges_fire_once_and_keep_their_history_position() -> None:
    policy = NudgePolicy(max_turns=10, progress_nudge_turn=2, turn_budget_warning=3)
    assert policy.apply([{"a": 1}], 1) == [{"a": 1}]
    assert policy.apply([{"a": 1}, {"b": 2}], 2) == [{"a": 1}, {"b": 2}]

    nudged = policy.apply([{"a": 1}, {"b": 2}, {"c": 3}], 3)
    assert len(nudged) == 4 and "state-changing workflow step" in nudged[3]["content"]
    again = policy.apply([{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}], 4)
    assert len(again) == 5 and again[3] == nudged[3] and again[4] == {"d": 4}

    policy.observe_tool_call("cases__create_from_order")
    assert policy.progress_seen
    warned = policy.apply([{}] * 7, 8)  # 7 turns used, 3 remain
    assert len(warned) == 9 and "3 of 10 turns remain" in warned[-1]["content"]
    assert len(policy.apply([{}] * 8, 9)) == 10  # nothing new


def test_progress_nudge_is_skipped_after_progress() -> None:
    policy = NudgePolicy(max_turns=10, progress_nudge_turn=1)
    policy.observe_tool_call("cm_intake.open_case")
    assert policy.apply([{}], 5) == [{}]


# ── tool guard ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_guard_contains_exceptions_and_records_timing(tmp_path: Path) -> None:
    stages = StageLog(tmp_path / "stages.jsonl")

    async def explode(_ctx: Any, _params: str) -> str:
        raise KeyError("path")

    tool = ToolGuard(stages=stages).wrap(
        SimpleNamespace(name="local-fs-read", on_invoke_tool=explode)
    )
    result = await tool.on_invoke_tool(None, "{}")
    assert result.startswith("ERROR: tool 'local-fs-read' failed: KeyError")
    assert stages.records[-1]["kind"] == "tool_call" and stages.records[-1]["ok"] is False
    assert stages.summary()["tool_errors"] == 1


def test_hook_tool_identity_translates_local_tools() -> None:
    assert hook_tool_identity("local-fs-read", {"path": "/h/a.md"}, mcp_name=None) == (
        "Read",
        {"file_path": "/h/a.md"},
    )
    assert hook_tool_identity("local-bash", {"command": "ls"}, mcp_name=None) == (
        "Bash",
        {"command": "ls"},
    )
    name, args = hook_tool_identity(
        "auth__submit_authorization", {"case_id": "C1"}, mcp_name="auth.submit_authorization"
    )
    assert name == "mcp__chi_bench__auth.submit_authorization" and args == {"case_id": "C1"}
    assert mech_common.canonical_tool_name(name, args) == "auth.submit_authorization"


@pytest.mark.asyncio
async def test_mechanism_gate_blocks_then_feeds_back(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    monkeypatch.delenv("CHI_BENCH_MECH_PREFLIGHT_TOOLS", raising=False)
    write_bundle(
        [MECHANISMS["preflight"]], state, state_dir=str(state), handbook_root=str(tmp_path)
    )
    gate = MechanismGate.from_state_dir(state, handbook_root=str(tmp_path))
    assert gate.names == ["preflight"] and "### Mechanism: preflight" in gate.prompt()
    assert {h.mechanism for h in gate.hooks} == {"preflight", "observe"}
    stages = StageLog(tmp_path / "stages.jsonl")
    calls: list[str] = []

    async def submit(_ctx: Any, params: str) -> str:
        calls.append(params)
        return '{"status": "submitted"}'

    tool = ToolGuard(stages=stages, gate=gate).wrap(
        SimpleNamespace(name="auth__submit_authorization", on_invoke_tool=submit),
        mcp_name="auth.submit_authorization",
    )

    blocked = await tool.on_invoke_tool(None, '{"case_id": "C1"}')
    assert blocked.startswith("[blocked by preflight]") and "preflight.json" in blocked
    assert calls == []
    assert stages.records[-1]["blocked"] is True

    (state / "preflight.json").write_text(
        json.dumps(
            {
                "tool": "auth.submit_authorization",
                "case_id": "C1",
                "checks": [{"item": "fields", "status": "pass", "evidence": "form"}],
                "blockers": [],
                "ready": True,
            }
        )
    )
    allowed = await tool.on_invoke_tool(None, '{"case_id": "C1"}')
    assert calls == ['{"case_id": "C1"}']
    assert allowed.startswith('{"status": "submitted"}')
    assert "[preflight]" in allowed and "Verify the resulting state" in allowed
    assert not (state / "preflight.json").exists()
    kinds = [r["kind"] for r in mech_common.load_jsonl(state / "events.jsonl")]
    # A denied call never executes, so (as in Claude Code) no PostToolUse
    # observer event is written for it.
    assert kinds == ["deny", "allow", "post_feedback", "tool_call"]


def test_gate_prompt_inlines_skills(tmp_path: Path) -> None:
    state = tmp_path / "state"
    write_bundle(resolve_mechanisms("policy_trace"), state, state_dir="/s", handbook_root="/h")
    gate = MechanismGate.from_state_dir(state, handbook_root="/h")
    assert gate.names == ["policy_trace"]
    prompt = gate.prompt()
    assert "### Mechanism: traceable policy use" in prompt
    assert "### Skill: policy-locator" in prompt and "/s/bin/policy_locate.py" in prompt
    assert "---\nname:" not in prompt  # frontmatter stripped
    # settings.json commands point at /s/bin; the gate resolves them under the
    # directory it was actually given.
    assert all(h.script.parent == state / "bin" for h in gate.hooks)


def test_gate_requires_uploaded_bundle(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="settings.json"):
        MechanismGate.from_state_dir(tmp_path / "missing")


# ── runner integration points ──────────────────────────────────────────


def test_trace_record_shows_requested_tool_name() -> None:
    class ToolCallItem:  # type name is what the runner dispatches on
        def __init__(self, raw: Any):
            self.raw_item = raw

    sink_call = ToolCallItem(SimpleNamespace(call_id="c2", name=UNKNOWN_TOOL_SINK, arguments="{}"))
    alias_call = ToolCallItem(SimpleNamespace(call_id="c1", name="local-fs-read", arguments="{}"))
    rewrites = {
        "c2": ToolCallRewrite("read_file", None, '{"path": "/y"}'),
        "c1": ToolCallRewrite("local_fs_read", "local-fs-read", "{}"),
    }
    record = openai_agents_runner._item_to_record(
        sink_call, {}, default_model="m", rewrites=rewrites
    )
    assert record["tool_calls"] == [
        {"id": "c2", "name": "read_file", "unresolved_tool": True, "arguments": {"path": "/y"}}
    ]
    record = openai_agents_runner._item_to_record(
        alias_call, {}, default_model="m", rewrites=rewrites
    )
    assert record["tool_calls"][0]["name"] == "local-fs-read"
    assert record["tool_calls"][0]["requested_name"] == "local_fs_read"


def test_write_result_carries_stage_summary_and_partial_usage(tmp_path: Path) -> None:
    openai_agents_runner.write_result(
        logs_dir=tmp_path,
        input_tokens=1,
        output_tokens=1,
        cached_tokens=0,
        cost_usd=0.1,
        turns=3,
        final_output=None,
        model="m",
        error="ModelBehaviorError: x",
        stages={"model_calls": 3},
        partial_usage={"input_tokens": 30, "output_tokens": 9, "cached_tokens": 0},
    )
    result = json.loads((tmp_path / "run_result.json").read_text())
    assert result["input_tokens"] is None and result["cost_usd"] is None
    assert result["stages"] == {"model_calls": 3}
    assert result["partial_usage"]["input_tokens"] == 30


# ── end-to-end against the pinned SDK ──────────────────────────────────


@pytest.mark.asyncio
async def test_sdk_run_survives_aliased_and_unknown_tool_calls(tmp_path: Path) -> None:
    agents = pytest.importorskip("agents")
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from chi_bench.experiment.agents.openai_agents_local_tools import (
        build_local_tools,
        build_unknown_tool_sink,
    )

    target = tmp_path / "note.txt"
    target.write_text("hello from disk\n")
    seen_inputs: list[list[Any]] = []

    class ScriptedModel(Model):
        def __init__(self) -> None:
            self.turn = 0

        async def get_response(self, system_instructions, input, *args, **kwargs):
            seen_inputs.append(list(input))
            self.turn += 1
            if self.turn == 1:
                output = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="c1",
                        name="local_fs_read",
                        arguments=json.dumps({"path": str(target)}),
                    ),
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="c2",
                        name="read_file",
                        arguments=json.dumps({"path": str(target)}),
                    ),
                    ResponseFunctionToolCall(
                        type="function_call", call_id="c3", name="local-fs-read", arguments="{bad"
                    ),
                ]
            else:
                output = [
                    ResponseOutputMessage(
                        id="m1",
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[
                            ResponseOutputText(type="output_text", text="done", annotations=[])
                        ],
                    )
                ]
            return ModelResponse(
                output=output,
                usage=Usage(requests=1, input_tokens=1, output_tokens=1),
                response_id=None,
            )

        def stream_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    stages = StageLog(tmp_path / "stages.jsonl")
    rewrites: dict[str, ToolCallRewrite] = {}
    guard = ToolGuard(stages=stages)
    tools = [guard.wrap(t) for t in build_local_tools(tmp_path)]
    tools.append(guard.wrap(build_unknown_tool_sink()))
    scripted = ScriptedModel()
    provider = ResilientModelProvider(
        SimpleNamespace(get_model=lambda _name: scripted),
        wrap=lambda inner: ResilientModel(inner, stages=stages, rewrites=rewrites),
    )
    agent = agents.Agent(name="t", instructions="x", tools=tools, model="scripted")

    result = await agents.Runner.run(
        agent,
        input="go",
        max_turns=3,
        run_config=agents.RunConfig(model_provider=provider, tracing_disabled=True),
    )

    assert result.final_output == "done"
    outputs = {
        item.raw_item["call_id"]: item.output
        for item in result.new_items
        if type(item).__name__ == "ToolCallOutputItem"
    }
    assert outputs["c1"] == "hello from disk\n"
    assert outputs["c2"].startswith("ERROR: tool 'read_file' does not exist")
    assert "local-fs-read" in outputs["c2"]  # suggestion
    assert outputs["c3"].startswith("ERROR: tool 'local-fs-read' failed")
    # The second model call sees the model's own names, not the sink.
    names = [i.get("name") for i in seen_inputs[1] if i.get("type") == "function_call"]
    assert names == ["local-fs-read", "read_file", "local-fs-read"]
    assert rewrites["c2"].unresolved and rewrites["c1"].requested == "local_fs_read"
    summary = stages.summary()
    assert summary["unknown_tool_calls"] == 1 and summary["alias_resolutions"] == 1
    assert summary["tool_calls"] == 3 and summary["tool_errors"] == 2


def test_resilience_constants_mirror_mechanisms_package() -> None:
    # The agent image ships the resilience module without the mechanisms
    # package, so the duplicated defaults must stay in sync.
    from chi_bench.experiment.agents import mechanisms, openai_agents_resilience as res

    assert res.DEFAULT_STATE_DIR == mechanisms.DEFAULT_STATE_DIR
    assert res.DEFAULT_HANDBOOK_ROOT == mechanisms.DEFAULT_HANDBOOK_ROOT
    asyncio.run(asyncio.sleep(0))


# ── continuation checks (empty reply / claim_done) ─────────────────────


def _msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


@pytest.mark.asyncio
async def test_empty_reply_is_continued_once(tmp_path: Path) -> None:
    stages = StageLog(tmp_path / "stages.jsonl")
    nudges = NudgePolicy(max_turns=10)
    claim = _fc("local-claim_done", "c1")
    inner = _InnerModel([_response(), _response(claim), _response(), _response()])
    model = ResilientModel(inner, stages=stages, rewrites={}, nudges=nudges)
    tools = [_tool("local-claim_done")]

    response = await model.get_response(
        None, [{"role": "user", "content": "go"}], None, tools, None, [], None
    )

    assert response.output == [claim]
    assert len(inner.calls) == 2
    nudge = inner.calls[1]["input"][-1]
    assert nudge["role"] == "user" and "no message and no tool call" in nudge["content"]
    assert [r["kind"] for r in stages.records] == ["model_call", "continuation", "model_call"]
    assert stages.records[-1]["continuation"] == "empty"
    assert nudges.claim_done_seen

    # Budget spent (default 1): a later empty reply is returned as-is.
    history = [{"role": "user", "content": "go"}, {}, {}]
    response = await model.get_response(None, history, None, tools, None, [], None)
    assert response.output == [] and len(inner.calls) == 3
    # The earlier notice stays in the history the model sees.
    assert any("no message and no tool call" in str(i) for i in inner.calls[2]["input"])


@pytest.mark.asyncio
async def test_final_text_without_claim_done_gets_completion_check(tmp_path: Path) -> None:
    stages = StageLog(tmp_path / "stages.jsonl")
    nudges = NudgePolicy(max_turns=10)
    nudges.observe_tool_call("determination__finalize")
    claim = _fc("local-claim_done", "c9")
    inner = _InnerModel([_response(_msg("All done.")), _response(claim)])
    model = ResilientModel(inner, stages=stages, rewrites={}, nudges=nudges)

    response = await model.get_response(
        None, [{"role": "user", "content": "go"}], None, [_tool("local-claim_done")], None, [], None
    )

    assert response.output == [claim]
    sent = inner.calls[1]["input"]
    assert sent[-2]["type"] == "message" and sent[-2]["content"][0].text == "All done."
    assert "without calling claim_done" in sent[-1]["content"]
    assert "determination.finalize" in sent[-1]["content"]
    assert stages.summary()["continuations"] == 1


@pytest.mark.asyncio
async def test_completion_check_lists_expected_terminal_actions_and_stops(tmp_path: Path) -> None:
    nudges = NudgePolicy(max_turns=10, completion_checks=1)
    inner = _InnerModel([_response(_msg("Done")), _response(_msg("Really done"))])
    model = ResilientModel(inner, stages=StageLog(tmp_path / "s.jsonl"), rewrites={}, nudges=nudges)

    response = await model.get_response(
        None, [{"role": "user", "content": "go"}], None, [], None, [], None
    )

    assert response.output[0].content[0].text == "Really done"
    notice = inner.calls[1]["input"][-1]["content"]
    assert "No terminal workflow action has been observed" in notice
    assert "cases.create_from_order" not in notice and "auth.submit_authorization" in notice


@pytest.mark.asyncio
async def test_text_reply_after_claim_done_ends_normally(tmp_path: Path) -> None:
    nudges = NudgePolicy(max_turns=10)
    nudges.observe_tool_call("local-claim_done")
    inner = _InnerModel([_response(_msg("Done"))])
    model = ResilientModel(inner, stages=StageLog(tmp_path / "s.jsonl"), rewrites={}, nudges=nudges)
    await model.get_response(None, [{"role": "user", "content": "go"}], None, [], None, [], None)
    assert len(inner.calls) == 1


@pytest.mark.asyncio
async def test_simultaneous_denies_become_one_templated_message(tmp_path: Path) -> None:
    from chi_bench.experiment.agents.openai_agents_resilience import HookOutcome

    denies = [
        HookOutcome("policy_trace", "PreToolUse", blocked=True, reason="cite the handbook"),
        HookOutcome("criteria_ledger", "PreToolUse", blocked=True, reason="write the ledger"),
    ]
    gate = SimpleNamespace(
        run_event=lambda event, name, inp, tool_response=None: (
            denies if event == "PreToolUse" else []
        )
    )

    async def never(_ctx: Any, _params: str) -> str:  # pragma: no cover
        raise AssertionError("blocked call must not execute")

    tool = ToolGuard(stages=StageLog(tmp_path / "s.jsonl"), gate=gate).wrap(
        SimpleNamespace(name="review__submit_md_case_decision", on_invoke_tool=never),
        mcp_name="review.submit_md_case_decision",
    )
    result = await tool.on_invoke_tool(None, "{}")
    assert result.startswith(
        "[blocked by policy_trace, criteria_ledger] `review__submit_md_case_decision` needs 2 records"
    )
    assert "(1) policy_trace:\ncite the handbook" in result
    assert "(2) criteria_ledger:\nwrite the ledger" in result
    assert "retry this same call once" in result


@pytest.mark.asyncio
async def test_sdk_run_continues_past_an_empty_reply(tmp_path: Path) -> None:
    agents = pytest.importorskip("agents")
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from chi_bench.experiment.agents.openai_agents_local_tools import build_local_tools

    class Scripted(Model):
        def __init__(self) -> None:
            self.calls = 0

        async def get_response(self, system_instructions, input, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                output: list[Any] = []  # reasoning-only reply: nothing visible
            elif self.calls == 2:
                output = [
                    ResponseFunctionToolCall(
                        type="function_call", call_id="c1", name="local-claim_done", arguments="{}"
                    )
                ]
            else:
                output = [
                    ResponseOutputMessage(
                        id="m1",
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[
                            ResponseOutputText(type="output_text", text="done", annotations=[])
                        ],
                    )
                ]
            return ModelResponse(
                output=output,
                usage=Usage(requests=1, input_tokens=3, output_tokens=1),
                response_id=None,
            )

        def stream_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    stages = StageLog(tmp_path / "stages.jsonl")
    nudges = NudgePolicy(max_turns=5)
    scripted = Scripted()
    provider = ResilientModelProvider(
        SimpleNamespace(get_model=lambda _name: scripted),
        wrap=lambda inner: ResilientModel(inner, stages=stages, rewrites={}, nudges=nudges),
    )
    agent = agents.Agent(name="t", instructions="x", tools=build_local_tools(tmp_path), model="s")
    result = await agents.Runner.run(
        agent,
        input="go",
        max_turns=5,
        run_config=agents.RunConfig(model_provider=provider, tracing_disabled=True),
    )

    assert result.final_output == "done"
    assert scripted.calls == 3 and len(result.raw_responses) == 2
    # The continuation call's usage is folded into the SDK-visible response.
    assert result.raw_responses[0].usage.input_tokens == 6
    assert stages.summary()["continuations"] == 1 and nudges.claim_done_seen
