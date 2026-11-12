"""Workspace judge runner for OpenAI-compatible models (vLLM, LiteLLM, OpenAI).

Drop-in for :class:`ClaudeAgentRunner` as used by :class:`WorkspaceJudge`: the
same ``run_session(prompt, cwd=..., allowed_tools=...)`` contract, where the
model reads the seeded workspace and writes ``verdicts.json`` itself. Instead
of the Claude Code CLI it runs a Chat Completions tool loop with file tools
that mirror Claude Code's (``Read``, ``Glob``, ``Grep``, ``Write``, ``Edit``),
confined to ``cwd`` and any ``add_dirs``.

Failures map onto the Claude runner's error types so the judge's existing
degradation contract applies unchanged: a timeout raises
``ClaudeTimeoutError``; transport or protocol failures raise
``ClaudeRunnerError``; a missing or malformed ``verdicts.json`` makes the trial
invalid exactly as with the CLI.

Controls: ``CHI_BENCH_JUDGE_MAX_TURNS`` (default 80),
``CHI_BENCH_JUDGE_READ_CHARS`` (per tool result, default 20000),
``CHI_BENCH_JUDGE_MAX_TOKENS`` (per model call, default 16384).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chi_bench.llm_routes import LLMRoute, LLMRouteError, chat_completion_raw, judge_route
from chi_bench.verifier.judge.claude_runner import ClaudeRunnerError, ClaudeTimeoutError

DEFAULT_MAX_TURNS = 80
DEFAULT_READ_CHARS = 20_000
DEFAULT_MAX_TOKENS = 16_384
MAX_LIST_RESULTS = 500
VERDICTS_FILE = "verdicts.json"
RUBRICS_FILE = "rubrics.json"
MAX_REPAIRS = 2

SYSTEM_PROMPT = """\
You are a grading agent working in a sandboxed directory: {cwd}
Use only the provided tools. Relative paths resolve against that directory; paths \
outside it are refused. Tool results longer than {read_chars} characters are cut, \
so page through large files with Read's offset and limit. Do the task in the user \
message, write verdicts.json with the Write or Edit tool, then reply with a short \
final message and no tool call."""

_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "Read": {
        "description": "Read a UTF-8 text file. Returns numbered lines. offset is the 1-indexed first line; limit is the number of lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["file_path"],
        },
    },
    "Glob": {
        "description": "List files matching a glob pattern (supports **), relative to path (default: working directory).",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    "Grep": {
        "description": "Search file contents with a regular expression. output_mode is files_with_matches (default) or content.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "output_mode": {"type": "string", "enum": ["files_with_matches", "content"]},
                "case_insensitive": {"type": "boolean"},
            },
            "required": ["pattern"],
        },
    },
    "Write": {
        "description": "Create or overwrite a file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"],
        },
    },
    "Edit": {
        "description": "Replace an exact string in a file. Fails if old_string is not unique unless replace_all is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


class _Sandbox:
    def __init__(self, cwd: Path, add_dirs: tuple[Path, ...], read_chars: int) -> None:
        self.cwd = cwd.resolve()
        self.roots = [self.cwd, *(d.resolve() for d in add_dirs)]
        self.read_chars = read_chars
        self.wrote_verdicts = False

    def path(self, raw: Any) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("a non-empty path is required")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        resolved = candidate.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.roots):
            raise ValueError(f"path is outside the working directory: {raw}")
        return resolved

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)

    def cap(self, text: str) -> str:
        if len(text) <= self.read_chars:
            return text
        return (
            text[: self.read_chars]
            + f"\n[truncated at {self.read_chars} of {len(text)} characters; "
            "use Read with offset and limit to see the rest]"
        )

    # -- tools -------------------------------------------------------------
    def Read(self, file_path: str, offset: int | None = None, limit: int | None = None) -> str:
        path = self.path(file_path)
        if not path.is_file():
            return f"ERROR: file not found: {file_path}"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(int(offset or 1), 1)
        end = len(lines) if limit is None else start - 1 + int(limit)
        body = "\n".join(f"{n:6}\t{lines[n - 1]}" for n in range(start, min(end, len(lines)) + 1))
        return self.cap(body or "(empty file or offset past end)")

    def Glob(self, pattern: str, path: str | None = None) -> str:
        root = self.path(path) if path else self.cwd
        matches = sorted(self.rel(p) for p in root.glob(pattern) if p.is_file() and self._inside(p))
        return self.cap("\n".join(matches[:MAX_LIST_RESULTS]) or "(no matches)")

    def Grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
        case_insensitive: bool = False,
    ) -> str:
        regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        root = self.path(path) if path else self.cwd
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        hits: list[str] = []
        for file in files:
            if not self._inside(file) or (glob and not fnmatch.fnmatch(file.name, glob)):
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if output_mode == "content":
                for n, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        hits.append(f"{self.rel(file)}:{n}:{line[:400]}")
            elif regex.search(text):
                hits.append(self.rel(file))
            if len(hits) >= MAX_LIST_RESULTS:
                break
        return self.cap("\n".join(hits) or "(no matches)")

    def Write(self, file_path: str, content: str) -> str:
        path = self.path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._note_write(path)
        return f"wrote {len(content)} characters to {self.rel(path)}"

    def Edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        path = self.path(file_path)
        if not path.is_file():
            return f"ERROR: file not found: {file_path}"
        text = path.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return "ERROR: old_string not found"
        if count > 1 and not replace_all:
            return f"ERROR: old_string matches {count} times; add context or set replace_all"
        path.write_text(
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1),
            encoding="utf-8",
        )
        self._note_write(path)
        return f"replaced {count if replace_all else 1} occurrence(s) in {self.rel(path)}"

    def _inside(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or root in resolved.parents for root in self.roots)

    def _note_write(self, path: Path) -> None:
        if path.name == VERDICTS_FILE:
            self.wrote_verdicts = True


def verdicts_problem(cwd: Path) -> str | None:
    """Describe what is wrong with ``verdicts.json`` in ``cwd``, or ``None`` if it is complete.

    Checks presence, JSON validity, that ``rubric_verdicts`` holds exactly the
    rubric IDs listed in ``rubrics.json``, and each entry's field types. It does
    not judge whether a verdict is right.
    """
    path = cwd / VERDICTS_FILE
    if not path.is_file():
        return (
            "You have not written verdicts.json yet. Write it now with the Write tool, "
            "following the output schema, then finish."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"verdicts.json is not valid JSON ({exc}). Rewrite it with the Write tool."
    verdicts = doc.get("rubric_verdicts") if isinstance(doc, dict) else None
    if not isinstance(verdicts, dict):
        return 'verdicts.json needs a top-level "rubric_verdicts" object. Rewrite it.'
    wanted: list[str] = []
    try:
        rubrics = json.loads((cwd / RUBRICS_FILE).read_text(encoding="utf-8"))
        wanted = [str(r["rubric_id"]) for r in rubrics.get("rubrics", []) if r.get("rubric_id")]
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        wanted = []
    problems: list[str] = []
    if wanted:
        missing = [rid for rid in wanted if rid not in verdicts]
        extra = [rid for rid in verdicts if rid not in wanted]
        if missing:
            problems.append(
                f"missing {len(missing)} rubric ID(s): {', '.join(missing)}. Read their full "
                "text in rubrics.json (page with Read offset/limit; Grep for the rubric_id) "
                "and grade them"
            )
        if extra:
            problems.append(f"remove ID(s) not in rubrics.json: {', '.join(extra)}")
    for rid, entry in verdicts.items():
        if not isinstance(entry, dict):
            problems.append(f"{rid}: entry must be an object")
            continue
        if not isinstance(entry.get("pass"), bool):
            problems.append(f'{rid}: "pass" must be true or false')
        if not str(entry.get("explanation") or "").strip():
            problems.append(f'{rid}: "explanation" is empty')
        refs = entry.get("evidence_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or not all(isinstance(r, str) and r.strip() for r in refs)
        ):
            problems.append(f'{rid}: "evidence_refs" must be a nonempty list of strings')
    if not problems:
        return None
    shown = problems[:12]
    more = f" (+{len(problems) - len(shown)} more)" if len(problems) > len(shown) else ""
    return (
        "verdicts.json is not complete yet: "
        + "; ".join(shown)
        + more
        + ". Fix verdicts.json with Write or Edit, then finish."
    )


class OpenAICompatJudgeRunner:
    """``run_session``-compatible judge runner over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        route: LLMRoute | None = None,
        *,
        timeout_seconds: int = 900,
        max_retries: int = 1,
        debug_dir: Path | None = None,
        on_stderr_line: Callable[[str], None] | None = None,
        max_turns: int | None = None,
        read_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.route = route or judge_route()
        if not self.route.is_openai:
            raise ValueError("OpenAICompatJudgeRunner needs an openai judge route")
        self.model = self.route.model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.debug_dir = debug_dir
        self.on_stderr_line = on_stderr_line
        self.max_turns = max_turns or _int_env("CHI_BENCH_JUDGE_MAX_TURNS", DEFAULT_MAX_TURNS)
        self.read_chars = read_chars or _int_env("CHI_BENCH_JUDGE_READ_CHARS", DEFAULT_READ_CHARS)
        self.max_tokens = max_tokens or _int_env("CHI_BENCH_JUDGE_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        self._invocation_count = 0

    def clone(
        self,
        *,
        debug_dir: Path | None = None,
        on_stderr_line: Callable[[str], None] | None = None,
    ) -> OpenAICompatJudgeRunner:
        return OpenAICompatJudgeRunner(
            self.route,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            debug_dir=self.debug_dir if debug_dir is None else debug_dir,
            on_stderr_line=self.on_stderr_line if on_stderr_line is None else on_stderr_line,
            max_turns=self.max_turns,
            read_chars=self.read_chars,
            max_tokens=self.max_tokens,
        )

    def run_session(
        self,
        prompt: str,
        *,
        cwd: Path,
        allowed_tools: tuple[str, ...] = ("Read", "Glob", "Grep", "Write", "Edit"),
        add_dirs: tuple[Path, ...] = (),
        mcp_config: dict[str, Any] | None = None,
        on_stderr_line: Callable[[str], None] | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        if mcp_config:
            raise ClaudeRunnerError("OpenAICompatJudgeRunner does not support MCP servers")
        self._invocation_count += 1
        budget = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        log = on_stderr_line or self.on_stderr_line
        last_error: ClaudeRunnerError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._session(prompt, cwd, allowed_tools, add_dirs, budget, attempt, log)
                return
            except ClaudeTimeoutError:
                raise
            except ClaudeRunnerError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2.0 * attempt)
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    def _session(
        self,
        prompt: str,
        cwd: Path,
        allowed_tools: tuple[str, ...],
        add_dirs: tuple[Path, ...],
        budget: int,
        attempt: int,
        log: Callable[[str], None] | None,
    ) -> None:
        sandbox = _Sandbox(cwd, add_dirs, self.read_chars)
        names = [name for name in allowed_tools if name in _TOOL_SPECS]
        tools = [
            {"type": "function", "function": {"name": name, **_TOOL_SPECS[name]}} for name in names
        ]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(cwd=sandbox.cwd, read_chars=self.read_chars),
            },
            {"role": "user", "content": prompt},
        ]
        meta: dict[str, Any] = {
            "runner": "openai_compat",
            **self.route.describe(),
            "attempt": attempt,
            "turns": 0,
            "tool_calls": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "status": "running",
        }
        started = time.monotonic()
        repairs = 0
        try:
            for turn in range(1, self.max_turns + 1):
                remaining = budget - (time.monotonic() - started)
                if remaining <= 1:
                    raise ClaudeTimeoutError(f"Judge session timed out after {budget}s.")
                try:
                    payload = chat_completion_raw(
                        self.route,
                        messages,
                        max_tokens=self.max_tokens,
                        tools=tools,
                        timeout_s=min(remaining, 600.0),
                        attempts=2,
                    )
                except LLMRouteError as exc:
                    if time.monotonic() - started >= budget - 1:
                        raise ClaudeTimeoutError(
                            f"Judge session timed out after {budget}s."
                        ) from exc
                    raise ClaudeRunnerError(str(exc)) from exc
                meta["turns"] = turn
                usage = payload.get("usage") or {}
                meta["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                meta["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                message = (payload.get("choices") or [{}])[0].get("message") or {}
                calls = message.get("tool_calls") or []
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                }
                if calls:
                    assistant["tool_calls"] = calls
                messages.append(assistant)
                if not calls:
                    # Format-only repair: the verdicts themselves are never changed
                    # here, and WorkspaceJudge still enforces the strict schema.
                    problem = verdicts_problem(sandbox.cwd)
                    if problem is None or repairs >= MAX_REPAIRS:
                        break
                    repairs += 1
                    meta["repairs"] = repairs
                    messages.append({"role": "user", "content": problem})
                    continue
                for call in calls:
                    result = self._dispatch(sandbox, call, names)
                    name = (call.get("function") or {}).get("name", "?")
                    meta["tool_calls"][name] = meta["tool_calls"].get(name, 0) + 1
                    if log:
                        log(f"[judge] turn {turn}: {name}")
                    messages.append(
                        {"role": "tool", "tool_call_id": call.get("id"), "content": result}
                    )
            else:
                raise ClaudeRunnerError(f"Judge session hit the {self.max_turns}-turn limit.")
            meta["status"] = "ok"
        except ClaudeTimeoutError:
            meta["status"] = "timeout"
            raise
        except ClaudeRunnerError as exc:
            meta["status"] = "error"
            meta["error"] = str(exc)[:500]
            raise
        finally:
            meta["duration_s"] = round(time.monotonic() - started, 2)
            meta["wrote_verdicts"] = sandbox.wrote_verdicts
            self._write_debug(prompt, messages, meta, attempt)

    @staticmethod
    def _dispatch(sandbox: _Sandbox, call: dict[str, Any], allowed: list[str]) -> str:
        function = call.get("function") or {}
        name = function.get("name")
        if name not in allowed:
            return f"ERROR: unknown tool {name!r}; available: {', '.join(allowed)}"
        raw = function.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError) as exc:
            return f"ERROR: tool arguments are not valid JSON: {exc}"
        if not isinstance(args, dict):
            return "ERROR: tool arguments must be a JSON object"
        try:
            return getattr(sandbox, name)(**args)
        except TypeError as exc:
            return f"ERROR: bad arguments for {name}: {exc}"
        except (ValueError, OSError, re.error) as exc:
            return f"ERROR: {exc}"

    def _write_debug(
        self, prompt: str, messages: list[dict[str, Any]], meta: dict[str, Any], attempt: int
    ) -> None:
        if self.debug_dir is None:
            return
        target = self.debug_dir / f"invocation_{self._invocation_count:03d}" / f"attempt_{attempt}"
        try:
            target.mkdir(parents=True, exist_ok=True)
            (target / "prompt.txt").write_text(prompt, encoding="utf-8")
            with (target / "transcript.jsonl").open("w", encoding="utf-8") as fh:
                for message in messages[1:]:
                    fh.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
            (target / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError:
            pass
