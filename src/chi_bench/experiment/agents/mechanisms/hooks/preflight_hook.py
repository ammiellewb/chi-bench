"""Mechanism 1 — executable preflight before terminal workflow actions.

PreToolUse: a terminal tool (submit / finalize / appeal / deliver / close)
is blocked unless ``preflight.json`` in the state dir names that tool, has
``ready: true``, and every check is ``pass`` or ``na`` with evidence.

PostToolUse: on success the preflight is archived to
``preflight_history.jsonl`` (so the next terminal action needs a fresh one)
and the model is asked to confirm the new state with a read-only call. On
an error response the preflight is kept and the model is told to repair
the input instead of moving on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from . import mech_common as mc
except ImportError:  # executed as a bare script inside the container
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mech_common as mc  # type: ignore[no-redef]

MECHANISM = "preflight"
PREFLIGHT_FILE = "preflight.json"
HISTORY_FILE = "preflight_history.jsonl"
POSTCHECK_FILE = "postcheck.jsonl"

# Generic, domain-wide terminal actions. Overridable at harness level via
# CHI_BENCH_MECH_PREFLIGHT_TOOLS (comma-separated ``ns.method`` names).
DEFAULT_TERMINAL_TOOLS: tuple[str, ...] = (
    # provider
    "auth.submit_authorization",
    "auth.finalize_provider_determination",
    "auth.respond_to_pend",
    "auth.file_appeal",
    "auth.file_external_appeal",
    # payer
    "payer_intake_hub.confirm_decisions",
    "review.submit_nurse_case_recommendation",
    "review.submit_md_case_decision",
    "determination.finalize",
    "payer_letter_center.deliver",
    # shared
    "p2p_session.submit_summary",
    # care management
    "cm_assessment.complete_assessment",
    "cm_care_plan.finalize",
    "cm_intake.disenroll_case",
)

# Read-only follow-ups suggested after a terminal action, by namespace.
VERIFY_HINTS: dict[str, str] = {
    "auth": "auth.check_status or cases.get_case",
    "payer_intake_hub": "payer_intake_hub.get_intake_case",
    "review": "review.get_md_review_data or review.get_nurse_review_data",
    "determination": "determination.get_summary",
    "payer_letter_center": "payer_letter_center.list / payer_letter_center.get",
    "p2p_session": "p2p_session.get_context",
    "cm_assessment": "cm_intake.get_case",
    "cm_care_plan": "cm_intake.get_case",
    "cm_intake": "cm_intake.get_case",
}

VALID_STATUS = {"pass", "fail", "na"}


def terminal_tools() -> tuple[str, ...]:
    import os

    raw = os.environ.get("CHI_BENCH_MECH_PREFLIGHT_TOOLS", "").strip()
    if not raw:
        return DEFAULT_TERMINAL_TOOLS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def validate_preflight(doc: Any, wanted_tool: str, call_args: dict[str, Any]) -> list[str]:
    """Return a list of problems; empty means the preflight clears the gate."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["preflight.json is missing or is not a JSON object"]
    tool = str(doc.get("tool") or "")
    if not mc.tool_matches(tool, wanted_tool) and tool != wanted_tool:
        problems.append(f"preflight.json is for tool {tool!r}, not {wanted_tool!r}")
    if doc.get("ready") is not True:
        problems.append('"ready" must be exactly true')
    checks = doc.get("checks")
    if not isinstance(checks, list) or not checks:
        problems.append('"checks" must be a non-empty list')
        checks = []
    for index, check in enumerate(checks, start=1):
        if not isinstance(check, dict):
            problems.append(f"check #{index} is not an object")
            continue
        status = str(check.get("status") or "").lower()
        item = str(check.get("item") or "").strip()
        evidence = str(check.get("evidence") or "").strip()
        if not item:
            problems.append(f"check #{index} has no item text")
        if status not in VALID_STATUS:
            problems.append(f"check #{index} ({item[:40]!r}) status must be pass|fail|na")
        elif status == "fail":
            problems.append(f"check #{index} ({item[:40]!r}) is still failing")
        elif status == "pass" and not evidence:
            problems.append(f"check #{index} ({item[:40]!r}) passes without evidence")
    blockers = doc.get("blockers")
    if isinstance(blockers, list) and blockers:
        problems.append(f"{len(blockers)} blocker(s) still listed: {blockers[:3]}")
    case_id = mc.extract_case_id(call_args)
    doc_case = doc.get("case_id")
    if case_id and isinstance(doc_case, str) and doc_case and doc_case != case_id:
        problems.append(f"preflight case_id {doc_case!r} != call case_id {case_id!r}")
    return problems


def _block_message(tool: str, problems: list[str], path: Path) -> str:
    bullet = "\n".join(f"  - {p}" for p in problems)
    return (
        f"[preflight] Blocked `{tool}`: a passing preflight is required before this "
        f"terminal action.\n{bullet}\n"
        f"Write {path} as JSON: "
        '{"tool": "<ns.method>", "case_id": "<id>", "checks": [{"item": "<what you '
        'verified>", "status": "pass|fail|na", "evidence": "<tool output or file that '
        'proves it>"}], "blockers": [], "ready": true}. '
        "Cover: required workflow steps completed in order, every required field and "
        "attachment present and validated, policy criteria evaluated with chart evidence, "
        "no pending peer-to-peer / pended request / outstanding info, and the action's "
        "target state is the intended one. Resolve every fail before retrying."
    )


def pre_tool_use(payload: dict[str, Any]) -> int:
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    canonical = mc.canonical_tool_name(tool_name, tool_input)
    wanted = mc.match_any(canonical, terminal_tools())
    if wanted is None:
        return 0
    path = mc.state_dir() / PREFLIGHT_FILE
    doc = mc.load_json(path)
    problems = validate_preflight(doc, wanted, mc.tool_args(tool_name, tool_input))
    if problems:
        return mc.deny(_block_message(wanted, problems, path), mechanism=MECHANISM, tool=wanted)
    return mc.allow(mechanism=MECHANISM, tool=wanted, note=f"{len(doc['checks'])} checks")


def post_tool_use(payload: dict[str, Any]) -> int:
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    canonical = mc.canonical_tool_name(tool_name, tool_input)
    wanted = mc.match_any(canonical, terminal_tools())
    if wanted is None:
        return 0
    response = payload.get("tool_response")
    state = mc.state_dir()
    path = state / PREFLIGHT_FILE
    errored = mc.looks_like_error(response)
    mc.append_jsonl(
        state / HISTORY_FILE,
        {
            "tool": wanted,
            "ok": not errored,
            "preflight": mc.load_json(path),
            "response_head": mc.response_text(response)[:600],
        },
    )
    if errored:
        return mc.post_feedback(
            f"[preflight] `{wanted}` returned an error. Do not proceed to the next step: "
            "read the message, fix the input or complete the missing prerequisite, update "
            f"{path} (set the relevant check back to fail until fixed), then retry.",
            mechanism=MECHANISM,
            tool=wanted,
            prompt=True,
        )
    try:
        path.unlink()
    except OSError:
        pass
    namespace = wanted.split(".", 1)[0]
    hint = VERIFY_HINTS.get(namespace, "the namespace's read-only get/status tool")
    return mc.post_feedback(
        f"[preflight] `{wanted}` succeeded. Verify the resulting state with {hint} before "
        f'continuing, and append {{"tool": "{wanted}", "observed_status": ...}} to '
        f"{state / POSTCHECK_FILE}. The consumed preflight was archived; a new "
        "preflight.json is required before the next terminal action.",
        mechanism=MECHANISM,
        tool=wanted,
        prompt=False,
    )


def main() -> int:
    payload = mc.read_payload()
    event = str(payload.get("hook_event_name") or "PreToolUse")
    if event == "PostToolUse":
        return post_tool_use(payload)
    return pre_tool_use(payload)


if __name__ == "__main__":
    sys.exit(main())
