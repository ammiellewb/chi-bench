"""Always-on observer (PostToolUse on every tool).

Writes one line per tool call to ``events.jsonl`` (tool, ok flag, case id)
and captures outreach turns to ``outreach_capture.jsonl`` so the consent
gate can verify quotes. Never blocks and never prompts.
"""

from __future__ import annotations

import sys
from typing import Any

try:
    from . import mech_common as mc
except ImportError:
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mech_common as mc  # type: ignore[no-redef]

MECHANISM = "observe"
CAPTURE_FILE = "outreach_capture.jsonl"
OUTREACH_TOOLS = ("cm_outreach.send_message", "cm_outreach.get_transcript")


def post_tool_use(payload: dict[str, Any]) -> int:
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    response = payload.get("tool_response")
    canonical = mc.canonical_tool_name(tool_name, tool_input)
    args = mc.tool_args(tool_name, tool_input)
    mc.log_event(
        "tool_call",
        mechanism=MECHANISM,
        tool=canonical or tool_name,
        native=tool_name,
        ok=not mc.looks_like_error(response),
        case_id=mc.extract_case_id(args),
    )
    if mc.match_any(canonical, OUTREACH_TOOLS):
        mc.append_jsonl(
            mc.state_dir() / CAPTURE_FILE,
            {"tool": canonical, "text": mc.response_text(response)[:20000]},
        )
    return 0


def main() -> int:
    payload = mc.read_payload()
    if str(payload.get("hook_event_name") or "") != "PostToolUse":
        return 0
    return post_tool_use(payload)


if __name__ == "__main__":
    sys.exit(main())
