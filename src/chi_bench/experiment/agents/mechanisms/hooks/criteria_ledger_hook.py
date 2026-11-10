"""Mechanism 3 — clinical reasoning separated from workflow execution.

PreToolUse on decision-bearing tools: ``criteria_ledger.json`` must exist and
carry a criterion-by-criterion assessment: each criterion has text, a
``met`` verdict in {yes, no, unknown}, chart evidence for yes/no, and an
explicit next action for unknown. A ledger written for a different case
blocks the call (criteria misapplication is the paper's top failure).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from . import mech_common as mc
except ImportError:
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mech_common as mc  # type: ignore[no-redef]

MECHANISM = "criteria_ledger"
LEDGER_FILE = "criteria_ledger.json"
VALID_MET = {"yes", "no", "unknown"}

# Final decisions only; intermediate saves are not gated (see policy_trace_hook).
DECISION_TOOLS: tuple[str, ...] = (
    "auth.submit_authorization",
    "auth.finalize_provider_determination",
    "auth.respond_to_pend",
    "auth.file_appeal",
    "auth.file_external_appeal",
    "review.submit_nurse_case_recommendation",
    "review.submit_md_case_decision",
    "determination.finalize",
    "cm_care_plan.finalize",
)


def decision_tools() -> tuple[str, ...]:
    import os

    raw = os.environ.get("CHI_BENCH_MECH_LEDGER_TOOLS", "").strip()
    if not raw:
        return DECISION_TOOLS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def validate_ledger(doc: Any, call_args: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["criteria_ledger.json is missing or is not a JSON object"]
    if not str(doc.get("decision") or "").strip():
        problems.append('"decision" (the question being decided) is empty')
    if not str(doc.get("conclusion") or "").strip():
        problems.append('"conclusion" is empty')
    criteria = doc.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        problems.append('"criteria" must be a non-empty list')
        criteria = []
    for index, item in enumerate(criteria, start=1):
        if not isinstance(item, dict):
            problems.append(f"criterion #{index} is not an object")
            continue
        label = str(item.get("id") or f"#{index}")
        if not str(item.get("criterion") or "").strip():
            problems.append(f"criterion {label}: no criterion text")
        if not str(item.get("source") or "").strip():
            problems.append(f"criterion {label}: no policy source")
        met = str(item.get("met") or "").lower()
        if met not in VALID_MET:
            problems.append(f"criterion {label}: met must be yes|no|unknown")
            continue
        evidence = item.get("evidence")
        has_evidence = isinstance(evidence, list) and any(
            isinstance(e, dict) and str(e.get("finding") or "").strip() for e in evidence
        )
        if met in {"yes", "no"} and not has_evidence:
            problems.append(f"criterion {label}: met={met} without chart evidence findings")
        if met == "unknown" and not str(item.get("action_if_unknown") or "").strip():
            problems.append(f"criterion {label}: unknown without action_if_unknown")
    case_id = mc.extract_case_id(call_args)
    doc_case = doc.get("case_id")
    if case_id and isinstance(doc_case, str) and doc_case and doc_case != case_id:
        problems.append(f"ledger case_id {doc_case!r} != call case_id {case_id!r}")
    return problems


def _block_message(tool: str, problems: list[str], path: Path) -> str:
    bullet = "\n".join(f"  - {p}" for p in problems)
    return (
        f"[criteria_ledger] Blocked `{tool}`: complete the criterion-by-criterion "
        f"assessment before executing a decision.\n{bullet}\n"
        f"Write {path} as JSON: "
        '{"case_id": "<id>", "decision": "<question>", "criteria": [{"id": "C1", '
        '"criterion": "<policy criterion text>", "source": "<handbook file/section>", '
        '"evidence": [{"where": "<chart section / document / transcript>", "finding": '
        '"<fact>"}], "met": "yes|no|unknown", "action_if_unknown": "<pend / request '
        'info / ask>"}], "conclusion": "<decision that follows from the criteria>"}. '
        "Assess every criterion the policy lists for this service and this patient, "
        "then execute only what the conclusion supports."
    )


def pre_tool_use(payload: dict[str, Any]) -> int:
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    canonical = mc.canonical_tool_name(tool_name, tool_input)
    wanted = mc.match_any(canonical, decision_tools())
    if wanted is None:
        return 0
    path = mc.state_dir() / LEDGER_FILE
    doc = mc.load_json(path)
    problems = validate_ledger(doc, mc.tool_args(tool_name, tool_input))
    if problems:
        return mc.deny(_block_message(wanted, problems, path), mechanism=MECHANISM, tool=wanted)
    return mc.allow(
        mechanism=MECHANISM, tool=wanted, note=f"{len(doc['criteria'])} criteria assessed"
    )


def main() -> int:
    payload = mc.read_payload()
    if str(payload.get("hook_event_name") or "PreToolUse") == "PostToolUse":
        return 0
    return pre_tool_use(payload)


if __name__ == "__main__":
    sys.exit(main())
