"""Harness-level resilience for the OpenAI Agents SDK runner.

Four concerns, all independent of the model vendor and all wired in by
``openai_agents_runner.run_agent``:

1. **Tool-name resilience.** Open-weight models routinely call ``local_fs_read``
   for ``local-fs-read`` or ``chart.get_patient`` for the sanitized
   ``chart__get_patient``. SDK 0.13.6 raises ``ModelBehaviorError`` on the
   first miss and the whole trial ends. :class:`ResilientModel` wraps the SDK
   model and, on every response, maps separator variants onto the registered
   tool name and routes anything still unknown to a hidden sink tool that
   returns an error *to the model* so the run continues.
2. **Tool-failure containment.** :class:`ToolGuard` wraps every FunctionTool so
   an exception inside a tool (invalid JSON arguments, a transport error on an
   MCP call) comes back as an ``ERROR:`` tool result instead of ending the run.
3. **Stage timing.** The same wrappers append one line per model call and per
   tool call to ``stages.jsonl`` *as they happen*, so a trial killed by the
   agent timeout still leaves a per-stage breakdown behind. ``run_result.json``
   carries the summary.
4. **Nudges and mechanisms.** :class:`NudgePolicy` injects an opt-in progress
   nudge (no state-changing call by turn N) and a turn-budget warning;
   :class:`MechanismGate` runs the ``claude-code-mechanisms`` hook scripts
   in-process (PreToolUse deny / PostToolUse feedback) for any
   OpenAI-compatible model. The gate reads the bundle the harness uploaded
   (``settings.json``, ``manifest.json``, ``prompt.md``) and never imports the
   mechanisms package, so the agent image keeps its closed set of source files.

This module imports neither the SDK nor any other chi-bench module at import
time; the runner's fake-SDK unit tests and the host-side harness tests keep
working without it.
"""

from __future__ import annotations

import asyncio
import copy
import difflib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:  # The SDK is only installed in the agent container.
    from agents.models.interface import Model as _ModelBase
    from agents.models.interface import ModelProvider as _ProviderBase
except Exception:  # pragma: no cover - exercised by the fake-SDK unit tests
    _ModelBase = object  # type: ignore[assignment,misc]
    _ProviderBase = object  # type: ignore[assignment,misc]

logger = logging.getLogger("openai_agents_resilience")

UNKNOWN_TOOL_SINK = "chi_bench__unknown_tool_sink"
STAGES_FILE = "stages.jsonl"
NUDGE_PREFIX = "[harness notice]"
MCP_HOOK_SERVER_ALIAS = "chi_bench"

# Mirrors ``mechanisms.components``; duplicated so the agent image needs no
# copy of that package (``test_runtime_assets`` pins the import closure).
DEFAULT_STATE_DIR = "/logs/agent/mechanisms"
DEFAULT_HANDBOOK_ROOT = "/workspace/skills/managed-care-operations-handbook"
SETTINGS_FILE = "settings.json"
MANIFEST_FILE = "manifest.json"
PROMPT_FILE = "prompt.md"

# State-changing calls that show the agent has moved past reading. One per
# domain entry point plus the decision-bearing calls, so a nudge fires only
# when the agent has done nothing but read. Override with
# ``OPENAI_AGENTS_PROGRESS_TOOLS`` (comma-separated ``ns.method``).
DEFAULT_PROGRESS_TOOLS: tuple[str, ...] = (
    # provider
    "cases.create_from_order",
    "auth.submit_authorization",
    "auth.respond_to_pend",
    "auth.file_appeal",
    # payer
    "payer_call_center.create_intake",
    "payer_intake_hub.save_servicing_facility",
    "payer_intake_hub.confirm_decisions",
    "triage.set_disposition",
    "review.save_criteria_evaluation",
    "review.submit_nurse_case_recommendation",
    "review.save_md_review",
    "review.submit_md_case_decision",
    "determination.finalize",
    "payer_letter_center.deliver",
    # care management
    "cm_intake.open_case",
    "cm_outreach.start_outreach_call",
    "cm_assessment.start_assessment",
    "cm_care_plan.create_draft",
)

# Terminal workflow actions (mirrors preflight_hook.DEFAULT_TERMINAL_TOOLS).
# Used by the completion check to tell the agent what has and has not happened.
DEFAULT_TERMINAL_TOOLS: tuple[str, ...] = (
    "auth.submit_authorization",
    "auth.finalize_provider_determination",
    "auth.respond_to_pend",
    "auth.file_appeal",
    "auth.file_external_appeal",
    "payer_intake_hub.confirm_decisions",
    "review.submit_nurse_case_recommendation",
    "review.submit_md_case_decision",
    "determination.finalize",
    "payer_letter_center.deliver",
    "p2p_session.submit_summary",
    "cm_assessment.complete_assessment",
    "cm_care_plan.finalize",
    "cm_intake.disenroll_case",
)
CLAIM_DONE_TOOL = "local-claim_done"
CLAIM_DONE_ACCEPTED = "task marked as done"

_COLLAPSE_RE = re.compile(r"[^a-z0-9]+")
_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def collapse_name(name: str) -> str:
    """``local-fs-read`` / ``local_fs_read`` / ``LOCAL.FS.READ`` -> ``localfsread``."""
    return _COLLAPSE_RE.sub("", name.lower())


def strip_mcp_prefix(name: str) -> str:
    """Drop a Claude-Code-style ``mcp__<server>__`` prefix a model may echo back."""
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3 and parts[2]:
            return parts[2]
    return name


def parse_tool_list(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return default
    return tuple(item.strip() for item in raw.replace(";", ",").split(",") if item.strip())


class ToolNameResolver:
    """Map a model-supplied tool name onto the registered name, if unambiguous."""

    def __init__(self, known_names: list[str]):
        self.known: set[str] = set(known_names)
        collapsed: dict[str, str | None] = {}
        for name in known_names:
            key = collapse_name(name)
            # Two registered tools with the same collapsed key are ambiguous;
            # keep them exact-match only.
            collapsed[key] = None if key in collapsed else name
        self._collapsed = collapsed

    def resolve(self, name: str) -> str | None:
        if name in self.known:
            return name
        for candidate in (name, strip_mcp_prefix(name)):
            resolved = self._collapsed.get(collapse_name(candidate))
            if resolved is not None:
                return resolved
        return None

    def suggestions(self, name: str, *, limit: int = 3) -> list[str]:
        pool = sorted(self.known)
        close = difflib.get_close_matches(name, pool, n=limit, cutoff=0.5)
        if close:
            return close
        # Fall back to shared words: ``read_file`` -> ``local-fs-read``.
        words = {w for w in _WORD_SPLIT_RE.split(strip_mcp_prefix(name).lower()) if len(w) > 2}
        scored = []
        for candidate in pool:
            overlap = words & {w for w in _WORD_SPLIT_RE.split(candidate.lower()) if w}
            if overlap:
                scored.append((-len(overlap), candidate))
        return [candidate for _score, candidate in sorted(scored)[:limit]]


@dataclass
class ToolCallRewrite:
    """What the model asked for versus what the SDK executed, keyed by call id."""

    requested: str
    resolved: str | None
    arguments: str

    @property
    def unresolved(self) -> bool:
        return self.resolved is None


class StageLog:
    """Append-only per-stage timing log plus an in-memory summary.

    Every record is flushed immediately so a hard kill (Harbor's agent
    timeout) still leaves the breakdown on disk.
    """

    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict[str, Any]] = []

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": round(time.time(), 3), "kind": kind, **fields}
        self.records.append(record)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - logging must never break a run
            logger.warning("Failed to append stage record: %s", exc)
        return record

    def partial_usage(self) -> dict[str, int]:
        """Token totals from the model calls that completed (an exception trial's floor)."""
        totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        for rec in self.records:
            if rec.get("kind") != "model_call" or rec.get("error"):
                continue
            for key in totals:
                value = rec.get(key)
                if isinstance(value, int):
                    totals[key] += value
        return totals

    def summary(self) -> dict[str, Any]:
        model_calls = [r for r in self.records if r["kind"] == "model_call"]
        tool_calls = [r for r in self.records if r["kind"] == "tool_call"]
        by_tool: dict[str, list[float]] = {}
        for rec in tool_calls:
            by_tool.setdefault(str(rec.get("tool")), []).append(float(rec.get("latency_sec", 0)))
        slowest = sorted(
            (
                {"tool": name, "calls": len(v), "total_sec": round(sum(v), 3)}
                for name, v in by_tool.items()
            ),
            key=lambda item: item["total_sec"],
            reverse=True,
        )[:5]
        return {
            "model_calls": len(model_calls),
            "model_sec": round(sum(float(r.get("latency_sec", 0)) for r in model_calls), 3),
            "model_errors": sum(1 for r in model_calls if r.get("error")),
            "tool_calls": len(tool_calls),
            "tool_sec": round(sum(float(r.get("latency_sec", 0)) for r in tool_calls), 3),
            "tool_errors": sum(1 for r in tool_calls if not r.get("ok", True)),
            "tool_blocked": sum(1 for r in tool_calls if r.get("blocked")),
            "alias_resolutions": sum(1 for r in self.records if r["kind"] == "tool_alias"),
            "unknown_tool_calls": sum(1 for r in self.records if r["kind"] == "unknown_tool"),
            "nudges": sum(1 for r in self.records if r["kind"] == "nudge"),
            "continuations": sum(1 for r in self.records if r["kind"] == "continuation"),
            "slowest_tools": slowest,
        }


# ---------------------------------------------------------------------------
# Nudges
# ---------------------------------------------------------------------------
@dataclass
class NudgePolicy:
    """Opt-in mid-run notices. Both thresholds default to off (0)."""

    max_turns: int
    progress_nudge_turn: int = 0
    turn_budget_warning: int = 0
    progress_tools: tuple[str, ...] = DEFAULT_PROGRESS_TOOLS
    # Continuation checks: never end a run on an empty reply, and ask once or
    # twice for claim_done before accepting a text-only final reply.
    empty_reply_retries: int = 1
    completion_checks: int = 2
    terminal_tools: tuple[str, ...] = DEFAULT_TERMINAL_TOOLS
    progress_seen: bool = False
    claim_done_seen: bool = False
    claim_done_calls: int = 0
    claim_done_accepted: bool = False
    terminal_seen: list[str] = field(default_factory=list)
    injected: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    _progress_sent: bool = False
    _budget_sent: bool = False
    _empty_sent: int = 0
    _completion_sent: int = 0

    def __post_init__(self) -> None:
        self._progress_keys = {collapse_name(name) for name in self.progress_tools}
        self._terminal_keys = {collapse_name(name): name for name in self.terminal_tools}
        self._claim_done_key = collapse_name(CLAIM_DONE_TOOL)

    def observe_tool_call(self, tool_name: str) -> None:
        key = collapse_name(strip_mcp_prefix(tool_name))
        if key in self._progress_keys:
            self.progress_seen = True
        if key == self._claim_done_key:
            self.claim_done_seen = True
        terminal = self._terminal_keys.get(key)
        if terminal and terminal not in self.terminal_seen:
            self.terminal_seen.append(terminal)

    def accept_claim_done(self) -> tuple[bool, str]:
        """Decide whether a ``claim_done`` call ends the run.

        Accepted on the first call once a terminal workflow action has been
        observed (or when no terminal set is configured); a premature first
        call gets one templated pushback, and a second call always ends the
        run so a model that keeps re-calling claim_done cannot loop (one Qwen
        trial called it 49 times and hit the turn cap).
        """
        self.claim_done_calls += 1
        if self.terminal_seen or not self.terminal_tools or self.claim_done_calls >= 2:
            self.claim_done_accepted = True
            return True, CLAIM_DONE_ACCEPTED
        return False, (
            f"{NUDGE_PREFIX} claim_done not accepted yet: no terminal workflow action has been "
            "observed, and this kind of workflow usually ends with one of: "
            + ", ".join(self.terminal_tools[:4])
            + ", …. Take the remaining steps (record the decision, finalize, generate and "
            "deliver required letters or notifications, submit the summary), then call "
            "claim_done again. A second claim_done always ends the run."
        )

    def continuation(self, kind: str) -> str | None:
        """Notice for a reply that would otherwise end the run; None once the budget is spent."""
        if kind == "empty":
            if self._empty_sent >= self.empty_reply_retries:
                return None
            self._empty_sent += 1
            return (
                f"{NUDGE_PREFIX} Your last reply contained no message and no tool call, so "
                "nothing happened. Continue the task now by calling the next tool. If the "
                "task is complete, call claim_done."
            )
        if self._completion_sent >= self.completion_checks:
            return None
        self._completion_sent += 1
        if self.terminal_seen:
            observed = "Terminal workflow actions observed so far: " + ", ".join(self.terminal_seen)
        else:
            observed = (
                "No terminal workflow action has been observed yet; this kind of workflow "
                "usually ends with one of: " + ", ".join(self.terminal_tools[:4]) + ", …"
            )
        return (
            f"{NUDGE_PREFIX} Your reply ended the turn without calling claim_done. {observed}. "
            "If required steps remain (for example recording the decision, finalizing the "
            "determination or plan, generating and delivering required letters or "
            "notifications, or submitting the summary), take them now. If the task is "
            "complete, call claim_done to finish."
        )

    def pending(self, turn: int) -> list[str]:
        """Notices to inject before model call ``turn`` (1-indexed)."""
        notices: list[str] = []
        if (
            self.progress_nudge_turn > 0
            and turn > self.progress_nudge_turn
            and not self.progress_seen
            and not self._progress_sent
        ):
            self._progress_sent = True
            examples = ", ".join(self.progress_tools[:4])
            notices.append(
                f"{NUDGE_PREFIX} You have used {turn - 1} of {self.max_turns} turns without a "
                "state-changing workflow step (for example "
                f"{examples}). Reading alone is not progress: open or create the case, request, "
                "or outreach session the task asks for now, then work through the remaining "
                "steps. Call claim_done only when the workflow is actually complete."
            )
        remaining = self.max_turns - (turn - 1)
        if (
            self.turn_budget_warning > 0
            and remaining <= self.turn_budget_warning
            and not self._budget_sent
        ):
            self._budget_sent = True
            notices.append(
                f"{NUDGE_PREFIX} {remaining} of {self.max_turns} turns remain. Finish the "
                "required workflow steps and call claim_done before the budget runs out; "
                "prefer state-changing actions over further reading."
            )
        return notices

    def apply(self, items: list[Any], turn: int) -> list[Any]:
        """Return ``items`` with every injected notice re-inserted at its original index."""
        for text in self.pending(turn):
            self.injected.append((len(items), {"role": "user", "content": text}))
        if not self.injected:
            return items
        out = list(items)
        for offset, (pos, item) in enumerate(sorted(self.injected, key=lambda p: p[0])):
            out.insert(min(pos + offset, len(out)), dict(item))
        return out


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
def _is_function_call(item: Any) -> bool:
    if isinstance(item, dict):
        return item.get("type") == "function_call"
    return getattr(item, "type", None) == "function_call"


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _set(item: Any, key: str, value: Any) -> None:
    if isinstance(item, dict):
        item[key] = value
    else:
        setattr(item, key, value)


def _as_input_item(output: Any) -> dict[str, Any] | None:
    """Convert a model output item into a history item (what the SDK's ``to_input_item`` does)."""
    if isinstance(output, dict):
        return dict(output)
    dump = getattr(output, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_unset=True)
        except Exception:  # pragma: no cover - defensive
            return None
    if hasattr(output, "__dict__"):
        return dict(vars(output))
    return None


def _message_text(response: Any) -> str:
    parts: list[str] = []
    for output in getattr(response, "output", None) or []:
        if (_get(output, "type") or "") != "message":
            continue
        for part in _get(output, "content") or []:
            text = _get(part, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _merge_usage(target: Any, source: Any) -> None:
    """Fold a continuation call's usage into the response the SDK will count."""
    target_usage = getattr(target, "usage", None)
    source_usage = getattr(source, "usage", None)
    add = getattr(target_usage, "add", None)
    if target_usage is None or source_usage is None or not callable(add):
        return
    try:
        add(source_usage)
    except Exception:  # pragma: no cover - usage shapes differ across SDK versions
        logger.debug("could not merge continuation usage", exc_info=True)


class ResilientModel(_ModelBase):  # type: ignore[misc,valid-type]
    """Wrap an SDK ``Model``: tool-name resolution, nudges, and stage timing."""

    def __init__(
        self,
        inner: Any,
        *,
        stages: StageLog,
        rewrites: dict[str, ToolCallRewrite],
        nudges: NudgePolicy | None = None,
    ):
        self.inner = inner
        self._stages = stages
        self._rewrites = rewrites
        self._nudges = nudges
        self.turn = 0
        self._last_input_len: int | None = None

    def __getattr__(self, name: str) -> Any:  # delegate anything we do not override
        return getattr(self.inner, name)

    def get_retry_advice(self, request: Any) -> Any:
        advice = getattr(self.inner, "get_retry_advice", None)
        return advice(request) if callable(advice) else None

    # -- input side ------------------------------------------------------
    def _begin_turn(self, items: list[Any]) -> int:
        # A retry of the same request carries an identical history length; a
        # new turn always appends at least the previous model output.
        if self._last_input_len is None or len(items) != self._last_input_len:
            self.turn += 1
        self._last_input_len = len(items)
        return self.turn

    def _restore_history(self, items: list[Any]) -> list[Any]:
        """Show the model its own unknown-tool call, not the sink it was routed to."""
        if not self._rewrites:
            return items
        out: list[Any] = []
        for item in items:
            if _is_function_call(item) and _get(item, "name") == UNKNOWN_TOOL_SINK:
                rewrite = self._rewrites.get(str(_get(item, "call_id")))
                if rewrite is not None:
                    item = copy.copy(item)
                    _set(item, "name", rewrite.requested)
                    _set(item, "arguments", rewrite.arguments)
            out.append(item)
        return out

    # -- output side -----------------------------------------------------
    def _rewrite_outputs(self, response: Any, resolver: ToolNameResolver) -> int:
        n_calls = 0
        for output in getattr(response, "output", None) or []:
            if not _is_function_call(output):
                continue
            n_calls += 1
            requested = str(_get(output, "name") or "")
            call_id = str(_get(output, "call_id") or _get(output, "id") or "")
            arguments = _get(output, "arguments")
            arguments_str = arguments if isinstance(arguments, str) else json.dumps(arguments or {})
            resolved = resolver.resolve(requested)
            if resolved == requested:
                if self._nudges is not None:
                    self._nudges.observe_tool_call(requested)
                continue
            if resolved is not None:
                _set(output, "name", resolved)
                self._rewrites[call_id] = ToolCallRewrite(requested, resolved, arguments_str)
                self._stages.record("tool_alias", requested=requested, resolved=resolved)
                if self._nudges is not None:
                    self._nudges.observe_tool_call(resolved)
                continue
            suggestions = resolver.suggestions(requested)
            _set(output, "name", UNKNOWN_TOOL_SINK)
            _set(
                output,
                "arguments",
                json.dumps(
                    {"name": requested, "arguments": arguments_str, "suggestions": suggestions}
                ),
            )
            self._rewrites[call_id] = ToolCallRewrite(requested, None, arguments_str)
            self._stages.record("unknown_tool", requested=requested, suggestions=suggestions)
        return n_calls

    # -- SDK interface ---------------------------------------------------
    async def get_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: Any = None,
        conversation_id: Any = None,
        prompt: Any = None,
        **kwargs: Any,
    ) -> Any:
        visible_tools = [t for t in tools if getattr(t, "name", None) != UNKNOWN_TOOL_SINK]
        resolver = ToolNameResolver(
            [t.name for t in visible_tools if isinstance(getattr(t, "name", None), str)]
        )
        items = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
        turn = self._begin_turn(items)
        raw_len = len(items)
        items = self._restore_history(items)
        if self._nudges is not None:
            before = len(self._nudges.injected)
            items = self._nudges.apply(items, turn)
            for _pos, notice in self._nudges.injected[before:]:
                self._stages.record("nudge", turn=turn, text=notice["content"])

        async def call(current: list[Any], continuation: str | None) -> Any:
            started = time.time()
            try:
                result = await self.inner.get_response(
                    system_instructions,
                    current,
                    model_settings,
                    visible_tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                    **kwargs,
                )
            except Exception as exc:
                self._stages.record(
                    "model_call",
                    turn=turn,
                    latency_sec=round(time.time() - started, 3),
                    input_items=len(current),
                    continuation=continuation,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
                raise
            latency = round(time.time() - started, 3)
            n_calls = self._rewrite_outputs(result, resolver)
            usage = getattr(result, "usage", None)
            details = getattr(usage, "input_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details is not None else None
            self._stages.record(
                "model_call",
                turn=turn,
                latency_sec=latency,
                input_items=len(current),
                n_tool_calls=n_calls,
                continuation=continuation,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                cached_tokens=int(cached) if isinstance(cached, int) else None,
            )
            return result

        response = await call(items, None)
        # A reply with no tool call ends the SDK run. Do not accept an empty
        # reply, and ask for claim_done before accepting a text-only one.
        while self._nudges is not None:
            kind = self._ending_kind(response)
            notice = self._nudges.continuation(kind) if kind else None
            if notice is None:
                break
            history = [it for it in map(_as_input_item, _get(response, "output") or []) if it]
            notice_item = {"role": "user", "content": notice}
            for item in history:
                self._nudges.injected.append((raw_len, item))
            self._nudges.injected.append((raw_len, notice_item))
            self._stages.record("continuation", turn=turn, reason=kind, text=notice)
            items = [*items, *history, notice_item]
            previous = response
            response = await call(items, kind)
            _merge_usage(response, previous)
        return response

    def _ending_kind(self, response: Any) -> str | None:
        """'empty' / 'final_text' when this reply would end the run, else None."""
        outputs = _get(response, "output") or []
        if any(_is_function_call(o) for o in outputs):
            return None
        if not _message_text(response).strip():
            return "empty"
        if self._nudges is not None and not self._nudges.claim_done_accepted:
            return "final_text"
        return None

    def stream_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # The runner uses Runner.run (non-streaming); streaming passes through
        # with the sink hidden but without name resolution.
        visible_tools = [t for t in tools if getattr(t, "name", None) != UNKNOWN_TOOL_SINK]
        return self.inner.stream_response(
            system_instructions, input, model_settings, visible_tools, *args, **kwargs
        )


class ResilientModelProvider(_ProviderBase):  # type: ignore[misc,valid-type]
    """Wrap every model the SDK resolves; cache per name so turn counters persist."""

    def __init__(
        self,
        inner: Any,
        *,
        wrap: Callable[[Any], ResilientModel],
        override_model: Any = None,
    ):
        self.inner = inner
        self._wrap = wrap
        self.override_model = override_model
        self._cache: dict[str | None, ResilientModel] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def get_model(self, model_name: str | None) -> ResilientModel:
        cached = self._cache.get(model_name)
        if cached is not None:
            return cached
        base = (
            self.override_model
            if self.override_model is not None
            else self.inner.get_model(model_name)
        )
        wrapped = self._wrap(base)
        self._cache[model_name] = wrapped
        return wrapped


# ---------------------------------------------------------------------------
# Mechanism gate: run the claude-code-mechanisms hook scripts in-process
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HookCommand:
    """One ``settings.json`` command hook, as Claude Code would run it."""

    mechanism: str
    event: str
    matcher: re.Pattern[str]
    script: Path
    timeout_sec: int = 30


@dataclass(frozen=True)
class HookOutcome:
    mechanism: str
    event: str
    blocked: bool = False
    reason: str | None = None
    context: str | None = None


# Local tools presented to the hook scripts under their Claude Code names so
# the policy_trace PostToolUse observer (``Read``/``Bash``) sees handbook reads.
_LOCAL_TOOL_HOOK_NAMES: dict[str, tuple[str, dict[str, str]]] = {
    "local-fs-read": ("Read", {"path": "file_path"}),
    "local-fs-write": ("Write", {"path": "file_path"}),
    "local-fs-edit": ("Edit", {"path": "file_path"}),
    "local-bash": ("Bash", {}),
}


def hook_tool_identity(
    tool_name: str, params: dict[str, Any], *, mcp_name: str | None
) -> tuple[str, dict[str, Any]]:
    """Translate a FunctionTool call into the ``(tool_name, tool_input)`` the hooks expect."""
    if mcp_name:
        return f"mcp__{MCP_HOOK_SERVER_ALIAS}__{mcp_name}", params
    mapping = _LOCAL_TOOL_HOOK_NAMES.get(tool_name)
    if mapping is None:
        return tool_name, params
    hook_name, renames = mapping
    return hook_name, {renames.get(key, key): value for key, value in params.items()}


def _mechanism_from_script(script: Path) -> str:
    return script.stem.removesuffix("_hook")


class MechanismGate:
    """Execute PreToolUse / PostToolUse hook scripts the way Claude Code would.

    Built from the bundle ``write_bundle`` produced on the host and the
    harness uploaded to ``state_dir``: ``settings.json`` supplies the hooks,
    ``manifest.json`` the mechanism names, ``prompt.md`` the system-prompt
    fragment (mechanism protocols plus inlined skills).
    """

    def __init__(
        self,
        hooks: list[HookCommand],
        *,
        names: list[str],
        prompt: str = "",
        state_dir: str = DEFAULT_STATE_DIR,
        handbook_root: str = DEFAULT_HANDBOOK_ROOT,
        python: str | None = None,
    ):
        self.hooks = list(hooks)
        self.names = list(names)
        self._prompt = prompt
        self.state_dir = state_dir
        self.handbook_root = handbook_root
        self.python = python or sys.executable

    @classmethod
    def from_state_dir(
        cls,
        state_dir: str | Path,
        *,
        handbook_root: str = DEFAULT_HANDBOOK_ROOT,
        python: str | None = None,
    ) -> MechanismGate:
        root = Path(state_dir)
        settings_path = root / SETTINGS_FILE
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"mechanisms requested but {settings_path} is missing or invalid; the harness "
                "uploads the hook bundle before the runner starts"
            ) from exc
        hooks: list[HookCommand] = []
        for event, entries in (settings.get("hooks") or {}).items():
            for entry in entries or []:
                matcher = re.compile(str(entry.get("matcher") or ".*"))
                for hook in entry.get("hooks") or []:
                    if hook.get("type", "command") != "command" or not hook.get("command"):
                        continue
                    script = Path(shlex.split(str(hook["command"]))[-1])
                    local = root / "bin" / script.name
                    # Commands name the configured state dir; prefer the
                    # directory we were actually given when it holds the script.
                    if not script.is_absolute() or (not script.exists() and local.exists()):
                        script = local
                    hooks.append(
                        HookCommand(
                            mechanism=_mechanism_from_script(script),
                            event=str(event),
                            matcher=matcher,
                            script=script,
                            timeout_sec=int(hook.get("timeout") or 30),
                        )
                    )
        names: list[str] = []
        try:
            manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
            names = [str(n) for n in manifest.get("mechanisms") or []]
        except (OSError, json.JSONDecodeError):
            pass
        try:
            prompt = (root / PROMPT_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            prompt = ""
        return cls(
            hooks,
            names=names,
            prompt=prompt,
            state_dir=str(root),
            handbook_root=handbook_root,
            python=python,
        )

    def prompt(self) -> str:
        return self._prompt

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CHI_BENCH_MECH_STATE_DIR"] = self.state_dir
        env["CHI_BENCH_MECH_HANDBOOK_ROOT"] = self.handbook_root
        return env

    def _run_script(
        self, script: Path, payload: dict[str, Any], timeout: int
    ) -> tuple[int, str, str]:
        proc = subprocess.run(
            [self.python, "-I", str(script)],
            input=json.dumps(payload, default=str),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env(),
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def run_event(
        self,
        event: str,
        hook_tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_response: Any = None,
    ) -> list[HookOutcome]:
        payload: dict[str, Any] = {
            "hook_event_name": event,
            "tool_name": hook_tool_name,
            "tool_input": tool_input,
            "session_id": "openai-agents",
            "cwd": os.getcwd(),
        }
        if event == "PostToolUse":
            payload["tool_response"] = tool_response
        outcomes: list[HookOutcome] = []
        for hook in self.hooks:
            if hook.event != event or not hook.matcher.search(hook_tool_name):
                continue
            try:
                code, stdout, stderr = self._run_script(hook.script, payload, hook.timeout_sec)
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("hook %s (%s) failed to run: %s", hook.script.name, event, exc)
                continue
            if event == "PreToolUse":
                if code == 2:
                    outcomes.append(
                        HookOutcome(hook.mechanism, event, blocked=True, reason=stderr.strip())
                    )
                elif code != 0:
                    logger.warning(
                        "hook %s exited %d: %s", hook.script.name, code, stderr.strip()[:300]
                    )
                continue
            for line in stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(doc, dict):
                    continue
                if doc.get("decision") == "block" and doc.get("reason"):
                    outcomes.append(HookOutcome(hook.mechanism, event, reason=str(doc["reason"])))
                specific = doc.get("hookSpecificOutput")
                if isinstance(specific, dict) and specific.get("additionalContext"):
                    outcomes.append(
                        HookOutcome(
                            hook.mechanism, event, context=str(specific["additionalContext"])
                        )
                    )
        return outcomes


# ---------------------------------------------------------------------------
# claim_done ends the run
# ---------------------------------------------------------------------------
def install_claim_done_policy(tool: Any, nudges: NudgePolicy) -> Any:
    """Make ``local-claim_done`` consult :meth:`NudgePolicy.accept_claim_done`."""

    async def on_invoke(_context: Any, _params: str) -> str:
        _accepted, message = nudges.accept_claim_done()
        return message

    tool.on_invoke_tool = on_invoke
    return tool


def build_claim_done_behavior(nudges: NudgePolicy) -> Callable[[Any, list[Any]], Any]:
    """``Agent(tool_use_behavior=...)`` hook: end the run once claim_done is accepted.

    The SDK keeps calling the model after any tool call, including claim_done,
    so a model that never writes a final message re-calls claim_done until the
    turn cap. Returning a final output here stops the loop at the accepted call.
    """

    def behavior(_context: Any, results: list[Any]) -> Any:
        from agents import ToolsToFinalOutputResult

        for result in results:
            name = getattr(getattr(result, "tool", None), "name", None)
            if name == CLAIM_DONE_TOOL and nudges.claim_done_accepted:
                return ToolsToFinalOutputResult(
                    is_final_output=True, final_output=str(getattr(result, "output", ""))
                )
        return ToolsToFinalOutputResult(is_final_output=False, final_output=None)

    return behavior


# ---------------------------------------------------------------------------
# Tool guard
# ---------------------------------------------------------------------------
def _result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return result["text"]
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def _result_ok(result: Any) -> bool:
    if isinstance(result, dict) and (result.get("isError") or result.get("is_error")):
        return False
    head = _result_text(result)[:200]
    return not head.startswith(("ERROR", "=== TIMEOUT", "[blocked"))


def _block_message(tool_name: str, denies: list[HookOutcome]) -> str:
    label = ", ".join(o.mechanism for o in denies)
    if len(denies) == 1:
        return f"[blocked by {label}] {denies[0].reason or ''}"
    sections = "\n\n".join(
        f"({i}) {o.mechanism}:\n{o.reason or ''}" for i, o in enumerate(denies, start=1)
    )
    return (
        f"[blocked by {label}] `{tool_name}` needs {len(denies)} records before it runs. "
        "Write every file described below (each section gives its JSON template), then "
        f"retry this same call once.\n\n{sections}"
    )


def _parse_params(params_str: str) -> dict[str, Any]:
    try:
        parsed = json.loads(params_str) if params_str else {}
    except (TypeError, ValueError):
        return {"_raw": params_str}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


class ToolGuard:
    """Wrap ``FunctionTool.on_invoke_tool`` with containment, timing, and gates."""

    def __init__(self, *, stages: StageLog, gate: MechanismGate | None = None):
        self.stages = stages
        self.gate = gate

    def wrap(self, tool: Any, *, mcp_name: str | None = None) -> Any:
        if getattr(tool, "_chi_bench_guarded", False):
            return tool
        original = tool.on_invoke_tool
        tool_name = str(getattr(tool, "name", "unknown_tool"))
        gate = self.gate
        stages = self.stages

        async def guarded(context: Any, params_str: str) -> Any:
            started = time.time()
            raw = params_str if isinstance(params_str, str) else json.dumps(params_str)
            params = _parse_params(raw)
            hook_name, hook_input = hook_tool_identity(tool_name, params, mcp_name=mcp_name)
            if gate is not None:
                try:
                    outcomes = await asyncio.to_thread(
                        gate.run_event, "PreToolUse", hook_name, hook_input
                    )
                except Exception as exc:  # pragma: no cover - hook runner must not end the run
                    logger.warning("PreToolUse gate failed for %s: %s", tool_name, exc)
                    outcomes = []
                denies = [o for o in outcomes if o.blocked]
                if denies:
                    stages.record(
                        "tool_call",
                        tool=tool_name,
                        latency_sec=round(time.time() - started, 3),
                        ok=False,
                        blocked=True,
                        mechanisms=[o.mechanism for o in denies],
                    )
                    return _block_message(tool_name, denies)
            try:
                result = await original(context, params_str)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = f"ERROR: tool '{tool_name}' failed: {type(exc).__name__}: {exc}"
            ok = _result_ok(result)
            if gate is not None:
                try:
                    outcomes = await asyncio.to_thread(
                        gate.run_event, "PostToolUse", hook_name, hook_input, tool_response=result
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("PostToolUse gate failed for %s: %s", tool_name, exc)
                    outcomes = []
                feedback = [f"[{o.mechanism}] {o.reason}" for o in outcomes if o.reason]
                feedback += [f"[{o.mechanism}] {o.context}" for o in outcomes if o.context]
                if feedback:
                    result = _result_text(result) + "\n\n" + "\n\n".join(feedback)
            stages.record(
                "tool_call",
                tool=tool_name,
                latency_sec=round(time.time() - started, 3),
                ok=ok,
            )
            return result

        tool.on_invoke_tool = guarded
        tool._chi_bench_guarded = True
        return tool
