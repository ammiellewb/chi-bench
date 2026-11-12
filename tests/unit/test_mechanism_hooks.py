"""Unit tests for the stdlib hook scripts behind `claude-code-mechanisms`."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chi_bench.experiment.agents.mechanisms.hooks import (
    completion_hook,
    consent_gate_hook,
    criteria_ledger_hook,
    mech_common,
    observe_hook,
    policy_locate,
    policy_trace_hook,
    preflight_hook,
)

HOOKS_DIR = Path(policy_trace_hook.__file__).parent


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state_dir = tmp_path / "state"
    monkeypatch.setenv("CHI_BENCH_MECH_STATE_DIR", str(state_dir))
    return state_dir


@pytest.fixture
def handbook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "handbook"
    (root / "references" / "payer-um").mkdir(parents=True)
    (root / "references" / "payer-um" / "bariatric.md").write_text(
        "# Bariatric surgery\n\n## Criteria\n\n"
        "Sleeve gastrectomy requires BMI of 40 or greater, or BMI of 35 with a comorbidity.\n"
        "A documented six month supervised weight loss program is required before approval.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHI_BENCH_MECH_HANDBOOK_ROOT", str(root))
    return root


def _mcp(tool: str, args: dict | None = None, event: str = "PreToolUse", response=None) -> dict:
    payload = {
        "hook_event_name": event,
        "tool_name": f"mcp__healthverse__{tool}",
        "tool_input": args or {},
        "session_id": "s1",
    }
    if event == "PostToolUse":
        payload["tool_response"] = response
    return payload


# ── mech_common ────────────────────────────────────────────────────────


def test_canonical_tool_name_handles_mcp_and_mcporter() -> None:
    assert (
        mech_common.canonical_tool_name("mcp__healthverse__auth.submit_authorization", {})
        == "auth.submit_authorization"
    )
    bash = {
        "command": 'npx mcporter call chi_bench determination.finalize --args \'{"case_id":"C1"}\''
    }
    assert mech_common.canonical_tool_name("Bash", bash) == "determination.finalize"
    assert mech_common.tool_args("Bash", bash) == {"case_id": "C1"}
    assert mech_common.canonical_tool_name("Read", {"file_path": "/x"}) is None


def test_tool_matches_accepts_alternate_separators() -> None:
    assert mech_common.tool_matches("auth__submit_authorization", "auth.submit_authorization")
    assert not mech_common.tool_matches("auth.check_status", "auth.submit_authorization")


def test_looks_like_error_detects_mcp_error_flags_and_text() -> None:
    assert mech_common.looks_like_error({"isError": True, "content": []})
    assert mech_common.looks_like_error([{"type": "text", "text": "ValueError: case not ready"}])
    assert not mech_common.looks_like_error({"status": "submitted", "id": "CASE-1"})


# ── preflight ──────────────────────────────────────────────────────────


def test_preflight_blocks_without_file(state: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = preflight_hook.pre_tool_use(_mcp("auth.submit_authorization", {"case_id": "C1"}))
    assert code == 2
    assert "preflight.json" in capsys.readouterr().err
    events = mech_common.load_jsonl(state / "events.jsonl")
    assert events[-1]["kind"] == "deny" and events[-1]["mechanism"] == "preflight"


def test_preflight_ignores_non_terminal_tools(state: Path) -> None:
    assert preflight_hook.pre_tool_use(_mcp("chart.search_patients")) == 0
    assert not (state / "events.jsonl").exists()


def _good_preflight(tool: str = "auth.submit_authorization", case: str = "C1") -> dict:
    return {
        "tool": tool,
        "case_id": case,
        "checks": [
            {"item": "bundle validated", "status": "pass", "evidence": "docs.validate ok"},
            {"item": "p2p pending", "status": "na", "evidence": ""},
        ],
        "blockers": [],
        "ready": True,
    }


def test_preflight_validation_rules() -> None:
    assert (
        preflight_hook.validate_preflight(_good_preflight(), "auth.submit_authorization", {}) == []
    )
    wrong_tool = preflight_hook.validate_preflight(
        _good_preflight("determination.finalize"), "auth.submit_authorization", {}
    )
    assert any("not 'auth.submit_authorization'" in p for p in wrong_tool)
    failing = _good_preflight()
    failing["checks"][0]["status"] = "fail"
    assert any(
        "still failing" in p
        for p in preflight_hook.validate_preflight(failing, "auth.submit_authorization", {})
    )
    no_evidence = _good_preflight()
    no_evidence["checks"][0]["evidence"] = ""
    assert any(
        "without evidence" in p
        for p in preflight_hook.validate_preflight(no_evidence, "auth.submit_authorization", {})
    )
    mismatch = preflight_hook.validate_preflight(
        _good_preflight(case="C9"), "auth.submit_authorization", {"case_id": "C1"}
    )
    assert any("case_id" in p for p in mismatch)


def test_preflight_allows_then_archives_on_success(
    state: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state.mkdir()
    (state / "preflight.json").write_text(json.dumps(_good_preflight()))
    assert preflight_hook.pre_tool_use(_mcp("auth.submit_authorization", {"case_id": "C1"})) == 0
    code = preflight_hook.post_tool_use(
        _mcp("auth.submit_authorization", {"case_id": "C1"}, "PostToolUse", {"status": "submitted"})
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert "additionalContext" in out["hookSpecificOutput"]
    assert "auth.check_status" in out["hookSpecificOutput"]["additionalContext"]
    assert not (state / "preflight.json").exists()
    history = mech_common.load_jsonl(state / "preflight_history.jsonl")
    assert history[0]["ok"] is True and history[0]["tool"] == "auth.submit_authorization"


def test_preflight_keeps_file_and_prompts_on_error(
    state: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state.mkdir()
    (state / "preflight.json").write_text(json.dumps(_good_preflight()))
    preflight_hook.post_tool_use(
        _mcp(
            "auth.submit_authorization",
            {},
            "PostToolUse",
            {"isError": True, "content": "ValueError"},
        )
    )
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "block" and "error" in out["reason"]
    assert (state / "preflight.json").exists()


def test_preflight_tool_set_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHI_BENCH_MECH_PREFLIGHT_TOOLS", "cm_care_plan.finalize, triage.route_case")
    assert preflight_hook.terminal_tools() == ("cm_care_plan.finalize", "triage.route_case")


# ── policy trace ───────────────────────────────────────────────────────


def test_policy_trace_blocks_empty_and_bad_citations(state: Path, handbook: Path, capsys) -> None:
    assert policy_trace_hook.pre_tool_use(_mcp("determination.finalize", {"case_id": "C1"})) == 2
    assert "no entries recorded" in capsys.readouterr().err
    state.mkdir(exist_ok=True)
    bad = {
        "criterion": "BMI threshold",
        "source_file": "references/payer-um/bariatric.md",
        "quote": "BMI of 50 or greater is required for everyone always",
        "interpretation": "x",
    }
    outside = dict(bad, source_file="/etc/passwd", quote="root:x:0:0:root:/root:/bin/bash")
    mech_common.append_jsonl(state / "policy_trace.jsonl", bad)
    mech_common.append_jsonl(state / "policy_trace.jsonl", outside)
    assert policy_trace_hook.pre_tool_use(_mcp("determination.finalize")) == 2
    err = capsys.readouterr().err
    assert "quote not found verbatim" in err and "not a file under" in err


def test_policy_trace_allows_verified_citation(state: Path, handbook: Path) -> None:
    state.mkdir(exist_ok=True)
    good = {
        "criterion": "BMI threshold",
        "source_file": str(handbook / "references/payer-um/bariatric.md"),
        "section": "Criteria",
        "quote": "requires BMI of 40   or greater, or BMI of 35 with a comorbidity",
        "interpretation": "BMI 42.3 satisfies the threshold",
    }
    mech_common.append_jsonl(state / "policy_trace.jsonl", good)
    assert policy_trace_hook.pre_tool_use(_mcp("determination.finalize")) == 0
    assert policy_trace_hook.pre_tool_use(_mcp("chart.search_patients")) == 0


def test_policy_trace_records_handbook_reads(state: Path, handbook: Path) -> None:
    read_payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Read",
        "tool_input": {"file_path": str(handbook / "references/payer-um/bariatric.md")},
        "tool_response": "...",
    }
    assert policy_trace_hook.post_tool_use(read_payload) == 0
    locate_payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {
            "command": 'python3 -I /logs/agent/mechanisms/bin/policy_locate.py "sleeve bmi"'
        },
        "tool_response": "...",
    }
    assert policy_trace_hook.post_tool_use(locate_payload) == 0
    reads = mech_common.load_jsonl(state / "policy_reads.jsonl")
    assert [r["read_kind"] for r in reads] == ["read", "locate"]


def test_policy_locate_finds_clause_and_prints_template(
    handbook: Path, state: Path, capsys
) -> None:
    assert policy_locate.main(["sleeve gastrectomy BMI comorbidity", "--max", "3"]) == 0
    out = capsys.readouterr().out
    assert "bariatric.md" in out and "policy_trace.jsonl entry template" in out
    assert policy_locate.main(["zzzz-unmatched-term"]) == 1


# ── criteria ledger ────────────────────────────────────────────────────


def _good_ledger(case: str = "C1") -> dict:
    return {
        "case_id": case,
        "decision": "Approve sleeve gastrectomy?",
        "criteria": [
            {
                "id": "C1",
                "criterion": "BMI >= 40",
                "source": "payer-um/bariatric.md#Criteria",
                "evidence": [{"where": "chart vitals", "finding": "BMI 42.3"}],
                "met": "yes",
            },
            {
                "id": "C2",
                "criterion": "6-month supervised program",
                "source": "payer-um/bariatric.md#Criteria",
                "evidence": [],
                "met": "unknown",
                "action_if_unknown": "pend for program documentation",
            },
        ],
        "conclusion": "pend",
    }


def test_criteria_ledger_validation_rules() -> None:
    assert criteria_ledger_hook.validate_ledger(_good_ledger(), {"case_id": "C1"}) == []
    missing_evidence = _good_ledger()
    missing_evidence["criteria"][0]["evidence"] = []
    assert any(
        "without chart evidence" in p
        for p in criteria_ledger_hook.validate_ledger(missing_evidence, {})
    )
    no_action = _good_ledger()
    no_action["criteria"][1]["action_if_unknown"] = ""
    assert any(
        "action_if_unknown" in p for p in criteria_ledger_hook.validate_ledger(no_action, {})
    )
    assert any(
        "case_id" in p
        for p in criteria_ledger_hook.validate_ledger(_good_ledger("C2"), {"case_id": "C1"})
    )
    assert criteria_ledger_hook.validate_ledger(None, {}) == [
        "criteria_ledger.json is missing or is not a JSON object"
    ]


def test_criteria_ledger_gate(state: Path, capsys) -> None:
    assert (
        criteria_ledger_hook.pre_tool_use(_mcp("review.submit_md_case_decision", {"case_id": "C1"}))
        == 2
    )
    assert "criteria_ledger.json" in capsys.readouterr().err
    state.mkdir(exist_ok=True)
    (state / "criteria_ledger.json").write_text(json.dumps(_good_ledger()))
    assert (
        criteria_ledger_hook.pre_tool_use(_mcp("review.submit_md_case_decision", {"case_id": "C1"}))
        == 0
    )


def test_decision_gates_skip_intermediate_saves(state: Path) -> None:
    # Drafting calls are never gated; only the final decision calls are.
    for tool in (
        "review.save_md_review",
        "review.save_criteria_evaluation",
        "triage.set_disposition",
        "cm_care_plan.save_care_plan",
    ):
        assert criteria_ledger_hook.pre_tool_use(_mcp(tool, {"case_id": "C1"})) == 0
        assert policy_trace_hook.pre_tool_use(_mcp(tool, {"case_id": "C1"})) == 0
    assert set(criteria_ledger_hook.DECISION_TOOLS) <= set(policy_trace_hook.DECISION_TOOLS)


# ── consent gate ───────────────────────────────────────────────────────


def _end_call(consent: bool, status: str = "engaged") -> dict:
    return _mcp(
        "cm_outreach.end_outreach_call",
        {"session_id": "S1", "summary": {"engagement_status": status, "consent_obtained": consent}},
    )


def test_consent_gate_allows_non_consent_summaries(state: Path) -> None:
    assert consent_gate_hook.pre_tool_use(_end_call(False, "refused")) == 0
    assert consent_gate_hook.pre_tool_use(_end_call(False, "tentative")) == 0


def test_consent_gate_blocks_unsupported_consent(state: Path, capsys) -> None:
    assert consent_gate_hook.pre_tool_use(_end_call(True)) == 2
    assert "consent_record.json" in capsys.readouterr().err
    state.mkdir(exist_ok=True)
    (state / "consent_record.json").write_text(
        json.dumps(
            {
                "consent_script_performed": True,
                "patient_affirmative_quote": "I'm not interested right now",
            }
        )
    )
    assert consent_gate_hook.pre_tool_use(_end_call(True)) == 2
    assert "reads as a refusal" in capsys.readouterr().err


def test_consent_gate_verifies_quote_against_capture(state: Path, capsys) -> None:
    state.mkdir(exist_ok=True)
    mech_common.append_jsonl(
        state / "outreach_capture.jsonl",
        {
            "tool": "cm_outreach.send_message",
            "text": json.dumps({"patient_response": "Yes, okay, sign me up for the program."}),
        },
    )
    (state / "consent_record.json").write_text(
        json.dumps(
            {
                "consent_script_performed": True,
                "patient_affirmative_quote": "Sure, that sounds great to me",
            }
        )
    )
    assert consent_gate_hook.pre_tool_use(_end_call(True)) == 2
    assert "does not appear in the captured" in capsys.readouterr().err
    (state / "consent_record.json").write_text(
        json.dumps(
            {"consent_script_performed": True, "patient_affirmative_quote": "Yes, okay, sign me up"}
        )
    )
    assert consent_gate_hook.pre_tool_use(_end_call(True)) == 0


def test_consent_gate_coaches_on_refusal(state: Path, capsys) -> None:
    payload = _mcp(
        "cm_outreach.send_message",
        {"session_id": "S1", "message": "..."},
        "PostToolUse",
        {"patient_response": "Honestly I'm not interested, I have enough on my plate."},
    )
    assert consent_gate_hook.post_tool_use(payload) == 0
    out = json.loads(capsys.readouterr().out)
    assert "do not probe" in out["hookSpecificOutput"]["additionalContext"]
    capsys.readouterr()
    payload["tool_response"] = {"patient_response": "Sure, tell me more."}
    assert consent_gate_hook.post_tool_use(payload) == 0
    assert capsys.readouterr().out == ""


# ── observer ───────────────────────────────────────────────────────────


def test_observer_logs_calls_and_captures_outreach(state: Path) -> None:
    observe_hook.post_tool_use(
        _mcp("cm_intake.get_case", {"case_id": "CM-1"}, "PostToolUse", {"status": "open"})
    )
    observe_hook.post_tool_use(
        _mcp(
            "cm_outreach.send_message",
            {"session_id": "S1"},
            "PostToolUse",
            {"patient_response": "hello"},
        )
    )
    events = mech_common.load_jsonl(state / "events.jsonl")
    assert events[0]["tool"] == "cm_intake.get_case" and events[0]["case_id"] == "CM-1"
    capture = mech_common.load_jsonl(state / "outreach_capture.jsonl")
    assert "hello" in capture[0]["text"]


# ── scripts run standalone exactly as Claude Code invokes them ─────────


@pytest.mark.parametrize(
    "script",
    [
        "preflight_hook.py",
        "policy_trace_hook.py",
        "criteria_ledger_hook.py",
        "consent_gate_hook.py",
        "observe_hook.py",
        "completion_hook.py",
    ],
)
def test_scripts_run_standalone_with_isolated_python(script: str, state: Path) -> None:
    payload = json.dumps(_mcp("auth.submit_authorization", {"case_id": "C1"}))
    proc = subprocess.run(
        [sys.executable, "-I", str(HOOKS_DIR / script)],
        input=payload,
        text=True,
        capture_output=True,
        env={"CHI_BENCH_MECH_STATE_DIR": str(state), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    expected = (
        2
        if script in {"preflight_hook.py", "policy_trace_hook.py", "criteria_ledger_hook.py"}
        else 0
    )
    assert proc.returncode == expected, proc.stderr
    if expected == 2:
        assert "Blocked" in proc.stderr


def test_read_payload_tolerates_garbage() -> None:
    assert mech_common.read_payload(io.StringIO("")) == {}
    assert mech_common.read_payload(io.StringIO("not json")) == {}
    assert mech_common.read_payload(io.StringIO("[1,2]")) == {}


def test_consent_gate_coaches_pacing_on_bundled_questions(state: Path, capsys) -> None:
    payload = _mcp(
        "cm_outreach.send_message",
        {"session_id": "S1", "message": "How is sleep? Any low mood? Drinking? Any falls?"},
        "PostToolUse",
        {"patient_response": "Uh, sleep is fine I guess."},
    )
    assert consent_gate_hook.post_tool_use(payload) == 0
    out = json.loads(capsys.readouterr().out)
    assert "one narrow question per turn" in out["hookSpecificOutput"]["additionalContext"]
    events = mech_common.load_jsonl(state / "events.jsonl")
    assert events[0]["kind"] == "pacing_detected" and "4 questions" in events[0]["phrase"]


def test_consent_gate_coaches_pacing_on_patient_overload(state: Path, capsys) -> None:
    payload = _mcp(
        "cm_outreach.send_message",
        {"session_id": "S1", "message": "Can I ask about your inhaler?"},
        "PostToolUse",
        {"patient_response": "Whew, okay, that's a lot at once. Let me think."},
    )
    assert consent_gate_hook.post_tool_use(payload) == 0
    out = json.loads(capsys.readouterr().out)
    assert "Pacing" in out["hookSpecificOutput"]["additionalContext"]
    capsys.readouterr()
    payload["tool_response"] = {"patient_response": "Sure, the morning one I take every day."}
    assert consent_gate_hook.post_tool_use(payload) == 0
    assert capsys.readouterr().out == ""


# ── completion (Stop hook) ─────────────────────────────────────────────


def _stop() -> dict:
    return {"hook_event_name": "Stop", "session_id": "s1", "stop_hook_active": False}


def _observe(state: Path, *tools: str, ok: bool = True) -> None:
    state.mkdir(parents=True, exist_ok=True)
    with (state / "events.jsonl").open("a") as fh:
        for tool in tools:
            fh.write(json.dumps({"kind": "tool_call", "tool": tool, "ok": ok}) + "\n")


def test_completion_blocks_stop_after_finalize_without_letters(state: Path, capsys) -> None:
    _observe(
        state, "p2p_session_submit_summary", "determination_finalize", "determination_get_summary"
    )
    assert completion_hook.stop(_stop()) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "block"
    assert "After `determination.finalize`, still missing" in out["reason"]
    assert "payer_letter_center.generate_approval or" in out["reason"]
    assert "payer_letter_center.deliver" in out["reason"]


def test_completion_allows_stop_when_letter_generated_and_delivered(state: Path, capsys) -> None:
    _observe(
        state,
        "determination_finalize",
        "payer_letter_center_generate_denial",
        "payer_letter_center_deliver",
    )
    assert completion_hook.stop(_stop()) == 0
    assert capsys.readouterr().out == ""


def test_completion_requires_delivery_and_ignores_failed_calls(state: Path, capsys) -> None:
    _observe(state, "determination_finalize", "payer_letter_center_generate_approval")
    _observe(state, "payer_letter_center_deliver", ok=False)
    completion_hook.stop(_stop())
    reason = json.loads(capsys.readouterr().out)["reason"]
    assert "payer_letter_center.deliver" in reason and "generate_approval" not in reason


def test_completion_does_nothing_without_a_trigger(state: Path, capsys) -> None:
    _observe(state, "triage_set_disposition", "review_save_md_review")
    assert completion_hook.stop(_stop()) == 0
    assert capsys.readouterr().out == ""


def test_completion_block_limit_prevents_loops(state: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("CHI_BENCH_MECH_COMPLETION_MAX_BLOCKS", raising=False)
    _observe(state, "determination_finalize")
    for _ in range(2):
        completion_hook.stop(_stop())
        assert json.loads(capsys.readouterr().out)["decision"] == "block"
    completion_hook.stop(_stop())
    assert capsys.readouterr().out == ""  # third stop is allowed
    kinds = [e["kind"] for e in mech_common.load_jsonl(state / "events.jsonl")]
    assert kinds.count("deny") == 2 and kinds[-1] == "allow"


def test_completion_ignores_subagent_stops(state: Path, capsys, monkeypatch) -> None:
    import io

    _observe(state, "determination_finalize")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"hook_event_name": "SubagentStop"})))
    assert completion_hook.main() == 0
    assert capsys.readouterr().out == ""


def test_preflight_success_feedback_points_to_remaining_steps(state: Path, capsys) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / "preflight.json").write_text("{}")
    preflight_hook.post_tool_use(
        _mcp("determination.finalize", {"case_id": "C1"}, "PostToolUse", {"status": "approved"})
    )
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "continue with the remaining workflow steps" in context
    assert "before continuing" not in context
