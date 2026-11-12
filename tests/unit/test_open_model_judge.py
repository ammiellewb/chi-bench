"""Open-model routes for the judge and simulators (no Anthropic key)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chi_bench import llm_routes
from chi_bench.llm_routes import LLMRouteError, chat_text, judge_route, simulator_route
from chi_bench.verifier.judge import openai_runner
from chi_bench.verifier.judge.claude_runner import (
    ClaudeAgentRunner,
    ClaudeRunnerError,
    ClaudeTimeoutError,
)
from chi_bench.verifier.judge.openai_runner import OpenAICompatJudgeRunner
from chi_bench.verifier.judge.runner_factory import judge_provenance, make_judge_runner
from chi_bench.verifier.judge.types import JudgeInput
from chi_bench.verifier.judge.workspace_judge import WorkspaceJudge

ROUTE_VARS = [
    "CHI_BENCH_JUDGE_PROVIDER",
    "CHI_BENCH_JUDGE_BASE_URL",
    "CHI_BENCH_JUDGE_API_KEY",
    "CHI_BENCH_JUDGE_MODEL",
    "CHI_BENCH_JUDGE_EXTRA_BODY",
    "CHI_BENCH_SIM_PROVIDER",
    "CHI_BENCH_SIM_BASE_URL",
    "CHI_BENCH_SIM_API_KEY",
    "CHI_BENCH_SIM_MODEL",
    "CHI_BENCH_SIM_EXTRA_BODY",
    "CHI_BENCH_SIM_MAX_TOKENS",
    "CHI_BENCH_P2P_SIMULATOR_MODEL",
    "CHI_BENCH_PATIENT_SIM_MODEL",
]
QWEN = "Qwen/Qwen3.6-35B-A3B-FP8"
EMPTY_VERDICTS = json.dumps({"rubric_verdicts": {}})
GPUBOX = "http://gpubox:8000/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ROUTE_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def open_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHI_BENCH_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("CHI_BENCH_JUDGE_BASE_URL", GPUBOX)
    monkeypatch.setenv("CHI_BENCH_JUDGE_MODEL", QWEN)


# ── routes ──────────────────────────────────────────────────────────────


def test_default_routes_stay_on_anthropic() -> None:
    assert judge_route().provider == "anthropic" and judge_route().model == "claude-opus-4-7"
    assert simulator_route("p2p", "CHI_BENCH_P2P_SIMULATOR_MODEL").provider == "anthropic"
    assert isinstance(make_judge_runner(timeout_seconds=10), ClaudeAgentRunner)
    assert judge_provenance() == {
        "judge_model": "claude-opus-4-7",
        "judge_provider": "anthropic",
        "judge_base_url": None,
    }


def test_openai_judge_requires_base_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHI_BENCH_JUDGE_PROVIDER", "openai")
    with pytest.raises(LLMRouteError, match="CHI_BENCH_JUDGE_BASE_URL"):
        judge_route()
    monkeypatch.setenv("CHI_BENCH_JUDGE_BASE_URL", GPUBOX)
    with pytest.raises(LLMRouteError, match="CHI_BENCH_JUDGE_MODEL"):
        judge_route()
    monkeypatch.setenv("CHI_BENCH_JUDGE_PROVIDER", "gemini")
    with pytest.raises(LLMRouteError, match="choose one of"):
        judge_route()


def test_simulators_follow_the_open_judge_route(open_judge, monkeypatch) -> None:
    monkeypatch.setenv("CHI_BENCH_JUDGE_EXTRA_BODY", '{"top_p": 0.9}')
    monkeypatch.setenv("CHI_BENCH_P2P_SIMULATOR_MODEL", "sonnet")  # Anthropic alias: ignored
    route = simulator_route("p2p", "CHI_BENCH_P2P_SIMULATOR_MODEL")
    assert (route.provider, route.model, route.base_url) == ("openai", QWEN, GPUBOX)
    assert route.extra_body == {"top_p": 0.9} and route.max_tokens == 4096
    monkeypatch.setenv("CHI_BENCH_P2P_SIMULATOR_MODEL", "other-model")
    assert simulator_route("p2p", "CHI_BENCH_P2P_SIMULATOR_MODEL").model == "other-model"
    assert isinstance(make_judge_runner(timeout_seconds=10), OpenAICompatJudgeRunner)
    assert judge_provenance()["judge_provider"] == "openai"


def test_explicit_sim_route_and_bad_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHI_BENCH_SIM_PROVIDER", "openai")
    monkeypatch.setenv("CHI_BENCH_SIM_BASE_URL", GPUBOX)
    with pytest.raises(LLMRouteError, match="CHI_BENCH_SIM_MODEL"):
        simulator_route("patient", "CHI_BENCH_PATIENT_SIM_MODEL")
    monkeypatch.setenv("CHI_BENCH_SIM_MODEL", QWEN)
    monkeypatch.setenv("CHI_BENCH_SIM_EXTRA_BODY", "[1]")
    with pytest.raises(LLMRouteError, match="JSON object"):
        simulator_route("patient", "CHI_BENCH_PATIENT_SIM_MODEL")
    # The judge stays on Anthropic when only the simulators move.
    assert judge_route().provider == "anthropic"


class _FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any]):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


def test_chat_text_strips_reasoning_and_sends_extra_body(open_judge, monkeypatch) -> None:
    monkeypatch.setenv(
        "CHI_BENCH_JUDGE_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}'
    )
    sent: list[dict[str, Any]] = []

    def fake_post(url, json, headers, timeout):
        sent.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(
            200,
            {
                "choices": [{"message": {"content": "<think>hmm</think> Yes, BMI is 31."}}],
                "usage": {"total_tokens": 42},
            },
        )

    monkeypatch.setattr(llm_routes.requests, "post", fake_post)
    route = simulator_route("p2p", "CHI_BENCH_P2P_SIMULATOR_MODEL")
    text, tokens = chat_text(route, "system", [{"role": "user", "content": "hi"}], max_tokens=64)
    assert (text, tokens) == ("Yes, BMI is 31.", 42)
    assert sent[0]["url"] == f"{GPUBOX}/chat/completions"
    assert sent[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent[0]["json"]["messages"][0] == {"role": "system", "content": "system"}
    assert sent[0]["headers"]["Authorization"] == "Bearer EMPTY"


def test_chat_text_reports_empty_replies_and_client_errors(open_judge, monkeypatch) -> None:
    replies = iter(
        [
            _FakeResponse(
                200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
            ),
            _FakeResponse(400, {"error": "bad model"}),
        ]
    )
    monkeypatch.setattr(llm_routes.requests, "post", lambda *a, **k: next(replies))
    route = judge_route()
    with pytest.raises(LLMRouteError, match="no text.*length"):
        chat_text(route, "s", [], max_tokens=8)
    with pytest.raises(LLMRouteError, match="HTTP 400"):
        chat_text(route, "s", [], max_tokens=8)


# ── judge runner ────────────────────────────────────────────────────────


def _call(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _reply(*calls: dict[str, Any], content: str = "") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _verdicts() -> dict[str, Any]:
    return {
        "rubric_verdicts": {
            "one": {
                "pass": True,
                "explanation": "The review cites the criterion.",
                "evidence_refs": ["agent_outputs/stage_review.json:criterion one"],
            },
            "two": {
                "pass": False,
                "explanation": "The finding is absent.",
                "evidence_refs": ["agent_outputs/stage_review.json:criterion two"],
            },
        },
        "session_metadata": {"investigation_notes": "Read rubrics and outputs."},
    }


def _script(monkeypatch: pytest.MonkeyPatch, replies: list[dict[str, Any]]) -> list[list[dict]]:
    seen: list[list[dict]] = []
    queue = iter(replies)

    def fake(route, messages, *, max_tokens, tools=None, timeout_s=300.0, attempts=3):
        seen.append([dict(m) for m in messages])
        assert {t["function"]["name"] for t in tools} == {"Read", "Glob", "Grep", "Write", "Edit"}
        return next(queue)

    monkeypatch.setattr(openai_runner, "chat_completion_raw", fake)
    return seen


def test_workspace_judge_end_to_end_on_open_runner(open_judge, monkeypatch, tmp_path: Path) -> None:
    seen = _script(
        monkeypatch,
        [
            _reply(
                _call("c1", "Read", file_path="rubrics.json"),
                _call("c2", "Glob", pattern="**/*.json"),
            ),
            _reply(_call("c3", "Grep", pattern="rubric_id", output_mode="content")),
            _reply(
                _call("c4", "Write", file_path="verdicts.json", content=json.dumps(_verdicts()))
            ),
            _reply(content="Done."),
        ],
    )
    judge_input = JudgeInput(
        "CASE-1",
        ({"rubric_id": "one"}, {"rubric_id": "two"}),
        {"case": "CASE-1"},
        (),
        {"stage_review": {"criteria": ["one"]}},
        (),
        "pa_um",
        "contract_v5",
    )
    runner = make_judge_runner(timeout_seconds=60, debug_dir=tmp_path / "debug")
    output = WorkspaceJudge(runner, timeout_seconds=60).judge(
        judge_input, workspace_dir=tmp_path / "ws"
    )

    assert output.judge_unavailable_reason is None
    assert output.verdicts["one"].passed is True and output.verdicts["two"].passed is False
    tool_results = [m for m in seen[1] if m["role"] == "tool"]
    assert '"rubric_id": "one"' in tool_results[0]["content"]  # Read returned rubrics.json
    assert "rubrics.json" in tool_results[1]["content"]  # Glob listed workspace files
    meta = json.loads(
        (tmp_path / "debug" / "invocation_001" / "attempt_1" / "metadata.json").read_text()
    )
    assert meta["status"] == "ok" and meta["wrote_verdicts"] and meta["provider"] == "openai"
    assert meta["tool_calls"] == {"Read": 1, "Glob": 1, "Grep": 1, "Write": 1}


def test_runner_refuses_paths_outside_the_workspace(
    open_judge, monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "secret.txt").write_text("expectations")
    ws = tmp_path / "ws"
    ws.mkdir()
    seen = _script(
        monkeypatch,
        [
            _reply(
                _call("c1", "Read", file_path="../secret.txt"),
                _call("c2", "Read", file_path=str(tmp_path / "secret.txt")),
            ),
            _reply(_call("c3", "Write", file_path="verdicts.json", content=EMPTY_VERDICTS)),
            _reply(content="done"),
        ],
    )
    OpenAICompatJudgeRunner(timeout_seconds=60).run_session("grade", cwd=ws)
    results = [m["content"] for m in seen[1] if m["role"] == "tool"]
    assert all("outside the working directory" in r for r in results)


def test_runner_repairs_missing_verdicts_at_most_twice(open_judge, monkeypatch, tmp_path) -> None:
    seen = _script(
        monkeypatch,
        [
            _reply(content="I think all pass."),
            _reply(content="Still no file."),
            _reply(content="no"),
        ],
    )
    OpenAICompatJudgeRunner(timeout_seconds=60).run_session("grade", cwd=tmp_path)
    assert len(seen) == 3 and "have not written verdicts.json" in seen[1][-1]["content"]


def test_runner_repairs_incomplete_verdicts(open_judge, monkeypatch, tmp_path) -> None:
    from chi_bench.verifier.judge.openai_runner import verdicts_problem

    (tmp_path / "rubrics.json").write_text(
        json.dumps({"rubrics": [{"rubric_id": "one"}, {"rubric_id": "two"}]})
    )
    partial = {"rubric_verdicts": {"one": _verdicts()["rubric_verdicts"]["one"]}}
    seen = _script(
        monkeypatch,
        [
            _reply(_call("c1", "Write", file_path="verdicts.json", content=json.dumps(partial))),
            _reply(content="done"),
            _reply(
                _call("c2", "Write", file_path="verdicts.json", content=json.dumps(_verdicts()))
            ),
            _reply(content="done"),
        ],
    )
    OpenAICompatJudgeRunner(timeout_seconds=60).run_session("grade", cwd=tmp_path)
    repair = seen[2][-1]["content"]
    assert "missing 1 rubric ID(s): two" in repair
    assert len(seen) == 4 and verdicts_problem(tmp_path) is None

    bad = {
        "rubric_verdicts": {
            "one": {"pass": "true", "explanation": "", "evidence_refs": []},
            "x": {},
        }
    }
    (tmp_path / "verdicts.json").write_text(json.dumps(bad))
    problem = verdicts_problem(tmp_path)
    assert "missing 1 rubric ID(s): two" in problem and "not in rubrics.json: x" in problem
    assert '"pass" must be true or false' in problem and '"explanation" is empty' in problem


def test_runner_tool_errors_go_back_to_the_model(open_judge, monkeypatch, tmp_path) -> None:
    (tmp_path / "a.txt").write_text("x\nx\n")
    seen = _script(
        monkeypatch,
        [
            _reply(
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{bad"}},
                _call("c2", "Bash", command="ls"),
                _call("c3", "Edit", file_path="a.txt", old_string="x", new_string="y"),
                _call("c4", "Read", file_path="a.txt", bogus=1),
            ),
            _reply(_call("c5", "Write", file_path="verdicts.json", content=EMPTY_VERDICTS)),
            _reply(content="ok"),
        ],
    )
    OpenAICompatJudgeRunner(timeout_seconds=60).run_session("grade", cwd=tmp_path)
    results = [m["content"] for m in seen[1] if m["role"] == "tool"]
    assert results[0].startswith("ERROR: tool arguments are not valid JSON")
    assert results[1].startswith("ERROR: unknown tool 'Bash'")
    assert "matches 2 times" in results[2]
    assert results[3].startswith("ERROR: bad arguments for Read")


def test_runner_caps_long_reads(open_judge, monkeypatch, tmp_path) -> None:
    (tmp_path / "big.md").write_text("line\n" * 5000)
    seen = _script(
        monkeypatch,
        [
            _reply(_call("c1", "Read", file_path="big.md")),
            _reply(_call("c2", "Write", file_path="verdicts.json", content=EMPTY_VERDICTS)),
            _reply(content="ok"),
        ],
    )
    OpenAICompatJudgeRunner(timeout_seconds=60, read_chars=500).run_session("g", cwd=tmp_path)
    content = [m for m in seen[1] if m["role"] == "tool"][0]["content"]
    assert len(content) < 700 and "use Read with offset and limit" in content


def test_runner_turn_limit_and_timeout(open_judge, monkeypatch, tmp_path) -> None:
    _script(monkeypatch, [_reply(_call(f"c{i}", "Glob", pattern="*")) for i in range(5)])
    with pytest.raises(ClaudeRunnerError, match="3-turn limit"):
        OpenAICompatJudgeRunner(timeout_seconds=60, max_turns=3).run_session("g", cwd=tmp_path)
    with pytest.raises(ClaudeTimeoutError):
        OpenAICompatJudgeRunner(timeout_seconds=60).run_session(
            "g", cwd=tmp_path, timeout_seconds=0
        )


def test_runner_transport_failure_is_a_runner_error(open_judge, monkeypatch, tmp_path) -> None:
    def boom(*args, **kwargs):
        raise LLMRouteError("connection refused")

    monkeypatch.setattr(openai_runner, "chat_completion_raw", boom)
    with pytest.raises(ClaudeRunnerError, match="connection refused"):
        OpenAICompatJudgeRunner(timeout_seconds=60, max_retries=1).run_session("g", cwd=tmp_path)


def test_clone_keeps_route_and_settings(open_judge, tmp_path) -> None:
    runner = OpenAICompatJudgeRunner(timeout_seconds=30, max_turns=7, debug_dir=tmp_path)
    clone = runner.clone(debug_dir=tmp_path / "vote_01")
    assert clone.route == runner.route and clone.max_turns == 7 and clone.timeout_seconds == 30
    assert clone.debug_dir == tmp_path / "vote_01"


# ── simulators ──────────────────────────────────────────────────────────


def test_p2p_simulator_uses_the_open_route_without_anthropic(open_judge, monkeypatch) -> None:
    from datetime import UTC, datetime

    import anthropic

    from chi_bench.core.models import P2PSessionRecord, PayerPeerToPeerRequest
    from chi_bench.services.p2p import P2PService

    def no_anthropic(*_a, **_k):
        raise AssertionError("Anthropic client must not be built on the openai route")

    calls: list[dict[str, Any]] = []

    def fake_chat_text(route, system, messages, *, max_tokens, timeout_s=300.0):
        calls.append(
            {"route": route, "system": system, "messages": messages, "max_tokens": max_tokens}
        )
        return "The addendum rules out concentric collapse.", 17

    monkeypatch.setattr(anthropic, "Anthropic", no_anthropic)
    monkeypatch.setattr(llm_routes, "chat_text", fake_chat_text)
    now = datetime(2026, 11, 12, tzinfo=UTC)
    request = PayerPeerToPeerRequest(id="p2p-1", case_id="case-1", created_at=now, updated_at=now)
    session = P2PSessionRecord(
        id="s-1",
        request_id="p2p-1",
        case_id="case-1",
        benchmark_side="payer",
        created_at=now,
        updated_at=now,
    )
    service = P2PService(ctx=None)
    monkeypatch.setattr(service, "_build_provider_context", lambda request: "You are the provider.")
    reply, fact = service._generate_counterpart_reply(
        session=session,
        request=request,
        contract={},
        message="Any new evidence?",
        agent_turn_index=1,
    )
    assert reply == "The addendum rules out concentric collapse." and fact is None
    assert calls[0]["route"].model == QWEN and calls[0]["system"] == "You are the provider."
    assert calls[0]["messages"] == [{"role": "user", "content": "Any new evidence?"}]
    assert calls[0]["max_tokens"] == 4096


async def test_patient_simulator_uses_the_open_route(open_judge, monkeypatch) -> None:
    from datetime import UTC, datetime

    from chi_bench.conversation import patient_simulator
    from chi_bench.conversation.message import STOP_SIGNAL
    from chi_bench.conversation.session import ConversationSession

    def no_anthropic(*_a, **_k):
        raise AssertionError("Anthropic client must not be built on the openai route")

    monkeypatch.setattr(patient_simulator.anthropic, "AsyncAnthropic", no_anthropic)
    monkeypatch.setattr(
        patient_simulator,
        "chat_text",
        lambda route, system, messages, *, max_tokens: (f"Okay. {STOP_SIGNAL}", 9),
    )
    simulator = patient_simulator.PatientSimulator(
        scenario={}, clinical_context="", patient_profile="Remain in character as the patient."
    )
    assert simulator._model == QWEN
    session = ConversationSession(
        session_id="s1",
        case_id="c1",
        patient_name="P",
        reason_for_call="outreach",
        created_at=datetime.now(UTC),
    )
    session.add_care_manager_message("Hello", datetime.now(UTC))
    response = await simulator.generate_response(session)
    assert response.content == "Okay." and response.stopped and response.token_usage == 9
