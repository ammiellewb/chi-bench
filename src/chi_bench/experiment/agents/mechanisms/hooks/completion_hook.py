"""Mechanism 5 — completion check before the agent stops.

Stop hook: when the agent tries to end the session, compare the tool calls
the observer logged (``events.jsonl``) against follow-on rules. Each rule is
"once TRIGGER has succeeded, each REQUIRED group must also have succeeded",
where a group is satisfied by any one of its tools. If a rule is unmet, the
stop is blocked with a templated reason naming the missing actions.

Rules are domain-generic tool names only (no task IDs or content). At most
``CHI_BENCH_MECH_COMPLETION_MAX_BLOCKS`` (default 2) stops are blocked per
session so a disagreement can never loop: after that the agent may stop.

Evidence: Sonnet 5.5 P2P trials that finalized the determination and then
stopped without generating or delivering the outcome letter failed only on
``outcome.letter_types``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    from . import mech_common as mc
except ImportError:  # executed as a bare script inside the container
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mech_common as mc  # type: ignore[no-redef]

MECHANISM = "completion"
BLOCKS_FILE = "completion_blocks.json"
DEFAULT_MAX_BLOCKS = 2

# (trigger, required groups, guidance). A group is satisfied by any one tool.
DEFAULT_RULES: tuple[tuple[str, tuple[tuple[str, ...], ...], str], ...] = (
    (
        "determination.finalize",
        (
            (
                "payer_letter_center.generate_approval",
                "payer_letter_center.generate_denial",
                "payer_letter_center.generate_pend",
            ),
            ("payer_letter_center.deliver",),
        ),
        "A finalized determination is normally followed by its outcome letter: generate the "
        "approval, denial or pend letter that matches the decision, deliver it, and send any "
        "member notification the handbook requires for this outcome.",
    ),
)


def max_blocks() -> int:
    raw = os.environ.get("CHI_BENCH_MECH_COMPLETION_MAX_BLOCKS", "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_BLOCKS
    except ValueError:
        return DEFAULT_MAX_BLOCKS


def succeeded_tools(events: list[Any]) -> list[str]:
    """Canonical names of tool calls the observer logged as successful, in order."""
    return [
        str(e.get("tool"))
        for e in events
        if isinstance(e, dict) and e.get("kind") == "tool_call" and e.get("ok") and e.get("tool")
    ]


def unmet_rules(called: list[str]) -> list[tuple[str, list[tuple[str, ...]], str]]:
    unmet = []
    for trigger, groups, guidance in DEFAULT_RULES:
        if not any(mc.tool_matches(name, trigger) for name in called):
            continue
        missing = [
            group for group in groups if not any(mc.match_any(name, group) for name in called)
        ]
        if missing:
            unmet.append((trigger, missing, guidance))
    return unmet


def block_reason(unmet: list[tuple[str, list[tuple[str, ...]], str]], remaining: int) -> str:
    lines = ["[completion] The workflow is not finished yet, so the session was not ended."]
    for trigger, missing, guidance in unmet:
        needed = "; ".join(" or ".join(group) for group in missing)
        lines.append(f"- After `{trigger}`, still missing: {needed}. {guidance}")
    lines.append(
        "Take these steps now, then finish. If the handbook says a step does not apply to this "
        "case, state that in your final message and stop again"
        + (
            f" (this check will not block more than {remaining} more time(s))."
            if remaining
            else "."
        )
    )
    return "\n".join(lines)


def stop(payload: dict[str, Any]) -> int:
    state = mc.state_dir()
    blocks_path = state / BLOCKS_FILE
    blocks = mc.load_json(blocks_path)
    count = int(blocks.get("count", 0)) if isinstance(blocks, dict) else 0
    limit = max_blocks()
    unmet = unmet_rules(succeeded_tools(mc.load_jsonl(state / mc.EVENTS_FILE)))
    if not unmet:
        return mc.allow(mechanism=MECHANISM, tool="Stop", note="workflow complete")
    if count >= limit:
        return mc.allow(mechanism=MECHANISM, tool="Stop", note=f"block limit {limit} reached")
    count += 1
    blocks_path.write_text(json.dumps({"count": count}), encoding="utf-8")
    reason = block_reason(unmet, limit - count)
    mc.log_event("deny", mechanism=MECHANISM, tool="Stop", reason=reason)
    # Stop hooks block with a top-level decision; the reason is shown to the model.
    sys.stdout.write(json.dumps({"decision": "block", "reason": reason}) + "\n")
    return 0


def main() -> int:
    payload = mc.read_payload()
    # Only the main agent's stop is checked; sub-agents legitimately end early.
    if str(payload.get("hook_event_name") or "") != "Stop":
        return 0
    return stop(payload)


if __name__ == "__main__":
    sys.exit(main())
