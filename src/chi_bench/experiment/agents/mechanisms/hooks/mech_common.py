"""Shared helpers for the chi-bench mechanism hooks (stdlib only).

Hook protocol (Claude Code ``settings.json`` command hooks):

* stdin  — one JSON object: ``{"hook_event_name", "tool_name", "tool_input",
  "tool_response"?, "session_id", "cwd", ...}``.
* PreToolUse — exit code 2 blocks the call and feeds stderr to the model.
* PostToolUse — stdout JSON ``{"decision": "block", "reason": ...}`` prompts
  the model with ``reason``; ``hookSpecificOutput.additionalContext`` adds
  context without a prompt.

State lives under ``$CHI_BENCH_MECH_STATE_DIR`` (default
``/logs/agent/mechanisms``), so every artifact is downloaded with the trial.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_STATE_DIR = "/logs/agent/mechanisms"
DEFAULT_HANDBOOK_ROOT = "/workspace/skills/managed-care-operations-handbook"
EVENTS_FILE = "events.jsonl"

_SEPARATORS = (".", "__", "-", "_")
_MCPORTER_RE = re.compile(r"\bmcporter\s+call\s+(\S+)\s+(\S+)")


def state_dir() -> Path:
    path = Path(os.environ.get("CHI_BENCH_MECH_STATE_DIR") or DEFAULT_STATE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def handbook_root() -> Path:
    return Path(os.environ.get("CHI_BENCH_MECH_HANDBOOK_ROOT") or DEFAULT_HANDBOOK_ROOT)


def read_payload(stream=None) -> dict[str, Any]:
    raw = (stream or sys.stdin).read()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def log_event(kind: str, **fields: Any) -> None:
    """Append one line to ``events.jsonl``; never raise from logging."""
    try:
        record = {"ts": time.time(), "kind": kind, **fields}
        with (state_dir() / EVENTS_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def canonical_tool_name(tool_name: str, tool_input: Any) -> str | None:
    """Return ``namespace.method`` for a chi-bench tool call, else ``None``.

    Handles native MCP names (``mcp__healthverse__auth.submit_authorization``)
    and the mcporter CLI form (``Bash`` with ``npx mcporter call chi_bench
    auth.submit_authorization ...``). Non-chi-bench tools return ``None``.
    """
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
        return None
    if tool_name == "Bash" and isinstance(tool_input, dict):
        command = str(tool_input.get("command") or "")
        match = _MCPORTER_RE.search(command)
        if match:
            return match.group(2)
    return None


def tool_matches(canonical: str | None, wanted: str) -> bool:
    """True if ``canonical`` names ``wanted`` (``ns.method``) under any separator."""
    if not canonical:
        return False
    namespace, method = wanted.split(".", 1)
    return any(canonical == f"{namespace}{sep}{method}" for sep in _SEPARATORS)


def match_any(canonical: str | None, wanted: tuple[str, ...] | list[str]) -> str | None:
    for name in wanted:
        if tool_matches(canonical, name):
            return name
    return None


def mcporter_args(tool_input: Any) -> dict[str, Any]:
    """Best-effort JSON args from an mcporter ``--args '{...}'`` Bash command."""
    if not isinstance(tool_input, dict):
        return {}
    command = str(tool_input.get("command") or "")
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {}
    for index, token in enumerate(tokens):
        if token == "--args" and index + 1 < len(tokens):
            try:
                parsed = json.loads(tokens[index + 1])
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def tool_args(tool_name: str, tool_input: Any) -> dict[str, Any]:
    """Structured arguments for the call regardless of MCP or CLI transport."""
    if tool_name == "Bash":
        return mcporter_args(tool_input)
    return tool_input if isinstance(tool_input, dict) else {}


def response_text(response: Any) -> str:
    """Flatten a tool_response (str / dict / content-block list) to text."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "text" in response and isinstance(response["text"], str):
            return response["text"]
        return json.dumps(response, default=str)
    if isinstance(response, list):
        return "\n".join(response_text(item) for item in response)
    return str(response)


_ERROR_RE = re.compile(
    r"(?i)(\w*error\b|\bexception\b|\btraceback\b|\binvalid\b|\bnot allowed\b|"
    r"\bfailed\b|\bforbidden\b)"
)


def looks_like_error(response: Any) -> bool:
    if isinstance(response, dict):
        if response.get("isError") is True or response.get("is_error") is True:
            return True
        if isinstance(response.get("error"), (str, dict)) and response["error"]:
            return True
    head = response_text(response)[:400]
    return bool(_ERROR_RE.search(head))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[Any] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_jsonl(path: Path, record: Any) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def extract_case_id(args: dict[str, Any]) -> str | None:
    for key in ("case_id", "caseId", "referral_id", "session_id"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# ── outcome helpers ─────────────────────────────────────────────────────


def deny(reason: str, *, mechanism: str, tool: str | None) -> int:
    """Block a PreToolUse call: log, write reason to stderr, return exit 2."""
    log_event("deny", mechanism=mechanism, tool=tool, reason=reason)
    sys.stderr.write(reason.rstrip() + "\n")
    return 2


def allow(*, mechanism: str, tool: str | None, note: str | None = None) -> int:
    log_event("allow", mechanism=mechanism, tool=tool, note=note)
    return 0


def post_feedback(reason: str, *, mechanism: str, tool: str | None, prompt: bool) -> int:
    """PostToolUse output: ``prompt=True`` makes the model respond to ``reason``."""
    log_event("post_feedback", mechanism=mechanism, tool=tool, reason=reason, prompt=prompt)
    if prompt:
        output: dict[str, Any] = {"decision": "block", "reason": reason}
    else:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": reason,
            }
        }
    sys.stdout.write(json.dumps(output) + "\n")
    return 0
