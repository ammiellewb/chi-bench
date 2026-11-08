"""Mechanism 2 — traceable policy use with a citation verifier.

PreToolUse on decision-bearing tools: ``policy_trace.jsonl`` must contain at
least one entry whose ``source_file`` resolves inside the handbook and whose
``quote`` appears verbatim (whitespace/case-insensitive) in that file. Bad
citations block the call and are listed back to the model.

PostToolUse on ``Read`` / ``Bash``: records handbook reads and
``policy_locate.py`` queries to ``policy_reads.jsonl`` so recall against a
task's expected policy files can be computed offline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    from . import mech_common as mc
except ImportError:
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mech_common as mc  # type: ignore[no-redef]

MECHANISM = "policy_trace"
TRACE_FILE = "policy_trace.jsonl"
READS_FILE = "policy_reads.jsonl"
MIN_QUOTE_CHARS = 20

DECISION_TOOLS: tuple[str, ...] = (
    # provider
    "auth.submit_authorization",
    "auth.finalize_provider_determination",
    "auth.respond_to_pend",
    "auth.file_appeal",
    "auth.file_external_appeal",
    # payer
    "triage.set_disposition",
    "review.save_criteria_evaluation",
    "review.submit_nurse_case_recommendation",
    "review.save_md_review",
    "review.submit_md_case_decision",
    "determination.finalize",
    # care management
    "cm_care_plan.save_care_plan",
    "cm_care_plan.finalize",
    "cm_intake.disenroll_case",
)

REQUIRED_FIELDS = ("criterion", "source_file", "quote", "interpretation")


def decision_tools() -> tuple[str, ...]:
    import os

    raw = os.environ.get("CHI_BENCH_MECH_POLICY_TOOLS", "").strip()
    if not raw:
        return DECISION_TOOLS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def resolve_source(source_file: str, root: Path) -> Path | None:
    """Resolve ``source_file`` (absolute or handbook-relative) inside ``root``."""
    candidate = Path(source_file)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError:
        return None
    if root_resolved not in resolved.parents and resolved != root_resolved:
        return None
    return resolved if resolved.is_file() else None


def verify_entry(entry: Any, root: Path, cache: dict[Path, str]) -> str | None:
    """Return a problem string or ``None`` when the citation verifies."""
    if not isinstance(entry, dict):
        return "entry is not a JSON object"
    missing = [field for field in REQUIRED_FIELDS if not str(entry.get(field) or "").strip()]
    if missing:
        return f"missing {', '.join(missing)}"
    quote = str(entry["quote"])
    if len(quote.strip()) < MIN_QUOTE_CHARS:
        return f"quote shorter than {MIN_QUOTE_CHARS} characters: {quote!r}"
    path = resolve_source(str(entry["source_file"]), root)
    if path is None:
        return f"source_file {entry['source_file']!r} is not a file under {root}"
    if path not in cache:
        try:
            cache[path] = mc.normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return f"could not read {path}"
    if mc.normalize_text(quote) not in cache[path]:
        return f"quote not found verbatim in {path.name}: {quote[:80]!r}"
    return None


def verify_trace(entries: list[Any], root: Path) -> tuple[int, list[str]]:
    cache: dict[Path, str] = {}
    problems: list[str] = []
    verified = 0
    for index, entry in enumerate(entries, start=1):
        problem = verify_entry(entry, root, cache)
        if problem is None:
            verified += 1
        else:
            problems.append(f"entry #{index}: {problem}")
    return verified, problems


def pre_tool_use(payload: dict[str, Any]) -> int:
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    canonical = mc.canonical_tool_name(tool_name, tool_input)
    wanted = mc.match_any(canonical, decision_tools())
    if wanted is None:
        return 0
    state = mc.state_dir()
    path = state / TRACE_FILE
    entries = mc.load_jsonl(path)
    root = mc.handbook_root()
    verified, problems = verify_trace(entries, root)
    if not entries or problems or verified == 0:
        detail = "\n".join(f"  - {p}" for p in problems) or "  - no entries recorded"
        return mc.deny(
            f"[policy_trace] Blocked `{wanted}`: every decision must cite the handbook "
            f"clause it rests on, and each citation must verify.\n{detail}\n"
            f"Append one JSON object per criterion to {path}: "
            '{"criterion": "<what is being decided>", "source_file": "<path under '
            f'{root}>", "section": "<heading>", "quote": "<verbatim sentence, '
            '>= 20 chars>", "interpretation": "<how it applies to this case>", '
            '"chart_evidence": "<fact from the chart/case>"}. '
            "Find clauses with: python3 -I "
            f'{state / "bin" / "policy_locate.py"} "<keywords>" '
            "then quote the printed sentence exactly. Fix or remove unverifiable entries.",
            mechanism=MECHANISM,
            tool=wanted,
        )
    return mc.allow(mechanism=MECHANISM, tool=wanted, note=f"{verified} citations verified")


_LOCATE_RE = re.compile(r"policy_locate\.py\s+(.*)$")


def post_tool_use(payload: dict[str, Any]) -> int:
    """Observe handbook reads; never blocks."""
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return 0
    root = str(mc.handbook_root())
    record: dict[str, Any] | None = None
    if tool_name == "Read":
        file_path = str(tool_input.get("file_path") or "")
        if file_path.startswith(root):
            record = {"read_kind": "read", "path": file_path}
    elif tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        match = _LOCATE_RE.search(command)
        if match:
            record = {"read_kind": "locate", "query": match.group(1)[:200]}
        elif root in command:
            record = {"read_kind": "shell", "command": command[:200]}
    if record is None:
        return 0
    mc.append_jsonl(mc.state_dir() / READS_FILE, record)
    mc.log_event("policy_read", mechanism=MECHANISM, **record)
    return 0


def main() -> int:
    payload = mc.read_payload()
    event = str(payload.get("hook_event_name") or "PreToolUse")
    if event == "PostToolUse":
        return post_tool_use(payload)
    return pre_tool_use(payload)


if __name__ == "__main__":
    sys.exit(main())
