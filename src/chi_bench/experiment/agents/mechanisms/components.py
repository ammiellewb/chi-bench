"""Toggleable harness mechanisms for A/B experiments on Claude Code.

Each :class:`Mechanism` is an independent component made of three parts:

* a system-prompt fragment that explains the protocol to the agent,
* Claude Code ``settings.json`` hooks (PreToolUse gates, PostToolUse
  observers) backed by the stdlib scripts in :mod:`.hooks`,
* optional skill files copied into ``$CLAUDE_CONFIG_DIR/skills``.

Mechanisms compose additively, so ``mechanisms=preflight`` and
``mechanisms=preflight,policy_trace`` differ by exactly one component.
The mapping to the alphaxiv plan: ``preflight`` → idea 1 (executable
workflow/safety preflight), ``policy_trace`` → idea 2 (traceable policy
use), ``criteria_ledger`` → idea 3 (reasoning separated from execution),
``consent_gate`` → idea 4 (autonomy as a constraint on claiming consent).
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

HOOKS_DIR = Path(__file__).parent / "hooks"
BIN_SUBDIR = "bin"
SKILLS_SUBDIR = "skills"
DEFAULT_STATE_DIR = "/logs/agent/mechanisms"
DEFAULT_HANDBOOK_ROOT = "/workspace/skills/managed-care-operations-handbook"

HookEvent = Literal["PreToolUse", "PostToolUse"]

# Chi-bench tools arrive either as native MCP tools (``mcp__<server>__ns.method``)
# or, in CLI mode, as ``Bash`` running ``npx mcporter call ...``. Scripts filter
# further; the matcher only narrows which calls spawn a hook process.
CHI_TOOL_MATCHER = r"^(mcp__.*|Bash)$"


@dataclass(frozen=True)
class HookSpec:
    event: HookEvent
    matcher: str
    script: str
    timeout_sec: int = 30


@dataclass(frozen=True)
class Mechanism:
    name: str
    summary: str
    prompt: str
    hooks: tuple[HookSpec, ...] = ()
    skills: dict[str, str] = field(default_factory=dict)


_PREFLIGHT_PROMPT = """\
### Mechanism: preflight before terminal actions
Terminal workflow actions (submit, finalize, appeal, respond-to-pend, deliver, \
complete-assessment, finalize care plan, disenroll, submit P2P summary) are gated. \
Before each one, write `{state}/preflight.json`:
{{"tool": "<ns.method>", "case_id": "<id>", "checks": [{{"item": "...", "status": \
"pass|fail|na", "evidence": "<tool output / file that proves it>"}}], "blockers": [], \
"ready": true}}
Checks must cover: prior workflow steps completed in the required order; every required \
field, form, and attachment present and validated; policy criteria evaluated against chart \
evidence; no pending peer-to-peer, pended request, or outstanding information; the action \
targets the intended case and outcome. The gate rejects any `fail`, any `pass` without \
evidence, and a preflight written for a different tool or case. After the action succeeds \
the preflight is archived; confirm the new state with a read-only call and append \
`{{"tool", "observed_status"}}` to `{state}/postcheck.jsonl`. If the action errors, fix the \
input before doing anything else."""

_POLICY_TRACE_PROMPT = """\
### Mechanism: traceable policy use
Decision-bearing tool calls are gated on verified handbook citations. Locate clauses with \
`python3 -I {state}/bin/policy_locate.py "<keywords>"` (optionally `--book <sub-book>`), \
read the surrounding section, then append one JSON object per criterion to \
`{state}/policy_trace.jsonl`:
{{"criterion": "...", "source_file": "<path under {handbook}>", "section": "<heading>", \
"quote": "<verbatim sentence, >= 20 chars>", "interpretation": "<how it applies here>", \
"chart_evidence": "<fact from the chart/case>"}}
The gate re-reads each cited file and rejects a quote that is not found verbatim, a path \
outside the handbook, or an empty trace. Cite the specific criterion clause, not the \
chapter title, and prefer the most specific sub-book for the role you are playing."""

_CRITERIA_LEDGER_PROMPT = """\
### Mechanism: assess criteria before executing
Separate clinical reasoning from workflow execution. Before any decision-bearing call \
(criteria evaluation, recommendation, MD decision, determination, submission, appeal, care \
plan), write `{state}/criteria_ledger.json`:
{{"case_id": "<id>", "decision": "<question>", "criteria": [{{"id": "C1", "criterion": \
"<policy criterion>", "source": "<handbook file/section>", "evidence": [{{"where": "<chart \
section / document / transcript>", "finding": "<fact>"}}], "met": "yes|no|unknown", \
"action_if_unknown": "<pend / request info / ask>"}}], "conclusion": "<decision the \
criteria support>"}}
List every criterion the policy applies to this service and patient, including exclusions \
and prerequisite steps. `yes`/`no` require at least one evidence finding; `unknown` requires \
a next action. The gate rejects an incomplete ledger or one written for another case. Then \
execute only what the conclusion supports, and update the ledger if new evidence appears."""

_CONSENT_GATE_PROMPT = """\
### Mechanism: patient autonomy in outreach
Engage autonomy-first: explain who you are and why you are calling, perform the consent \
script, and ask for an explicit yes before enrolling. If the patient hesitates or declines, \
acknowledge it in one sentence, do not probe for concerns or re-pitch, offer at most one \
low-commitment option, and accept a no. Consent to enroll is not consent to be screened: \
after enrollment ask one narrow question per turn, explain its relevance, ask permission \
before each new sensitive topic, and remind the patient they can skip or stop. Never bundle \
several questions in one message. Keep `{state}/consent_record.json` current:
{{"session_id": "<id>", "consent_script_performed": true|false, "patient_affirmative_quote": \
"<patient's exact words agreeing>", "refusals_or_hesitations": ["<exact words>"], \
"response_to_refusal": "<how you acknowledged it>"}}
Ending a call with `consent_obtained: true` or `engagement_status: engaged` is gated on this \
record: the quote must be the patient's own affirmative words from the transcript. Without \
an explicit agreement, record `tentative` or `refused` with `consent_obtained: false`."""

_POLICY_LOCATOR_SKILL = """\
---
name: policy-locator
description: Find and cite the managed-care handbook clause that governs a decision before acting on it.
---

# Policy locator

Use this whenever a decision depends on a coverage rule, review criterion, SLA, routing
rule, letter requirement, or care-management protocol.

1. Search: `python3 -I {state}/bin/policy_locate.py "<3-6 keywords>" [--book <sub-book>]`
   The handbook lives at `{handbook}`; sub-books are the directories under `references/`.
2. Open the top hit with Read and confirm the sentence that actually states the rule.
3. Append a verified entry to `{state}/policy_trace.jsonl` (schema in the system prompt).
   Quote one sentence verbatim; the gate re-checks it against the file.
4. Interpret the clause for this case in one sentence and tie it to a chart fact.

Cite the specific criterion, exclusion, or step, not a chapter heading. If two clauses
conflict, cite both and state which one controls and why.
"""


MECHANISMS: dict[str, Mechanism] = {
    "preflight": Mechanism(
        name="preflight",
        summary="Gate terminal actions on a passing readiness checklist; verify state after.",
        prompt=_PREFLIGHT_PROMPT,
        hooks=(
            HookSpec("PreToolUse", CHI_TOOL_MATCHER, "preflight_hook.py"),
            HookSpec("PostToolUse", CHI_TOOL_MATCHER, "preflight_hook.py"),
        ),
    ),
    "policy_trace": Mechanism(
        name="policy_trace",
        summary="Require verified handbook citations before decision-bearing calls.",
        prompt=_POLICY_TRACE_PROMPT,
        hooks=(
            HookSpec("PreToolUse", CHI_TOOL_MATCHER, "policy_trace_hook.py"),
            HookSpec("PostToolUse", r"^(Read|Bash)$", "policy_trace_hook.py"),
        ),
        skills={"policy-locator/SKILL.md": _POLICY_LOCATOR_SKILL},
    ),
    "criteria_ledger": Mechanism(
        name="criteria_ledger",
        summary="Require a criterion-by-criterion assessment before executing a decision.",
        prompt=_CRITERIA_LEDGER_PROMPT,
        hooks=(HookSpec("PreToolUse", CHI_TOOL_MATCHER, "criteria_ledger_hook.py"),),
    ),
    "consent_gate": Mechanism(
        name="consent_gate",
        summary="Coach autonomy-first outreach; gate claimed consent on a transcript quote.",
        prompt=_CONSENT_GATE_PROMPT,
        hooks=(
            HookSpec("PreToolUse", CHI_TOOL_MATCHER, "consent_gate_hook.py"),
            HookSpec("PostToolUse", CHI_TOOL_MATCHER, "consent_gate_hook.py"),
        ),
    ),
}

OBSERVER = Mechanism(
    name="observe",
    summary="Log every tool call and capture outreach turns; no behavior change.",
    prompt="",
    hooks=(HookSpec("PostToolUse", r".*", "observe_hook.py", timeout_sec=15),),
)

ALL_ALIAS = "all"
PROMPT_HEADER = (
    "## chi-bench workflow mechanisms\n"
    "The mechanisms below are enforced by tool hooks. A blocked call returns the reason; "
    "satisfy it and retry rather than working around the gate. Keep every record under "
    "`{state}` accurate: they are part of your work product."
)


def resolve_mechanisms(spec: str | None) -> list[Mechanism]:
    """Parse ``"preflight,policy_trace"`` / ``"all"`` / ``""`` into components.

    Unknown names raise ``ValueError`` so a typo in a matrix row fails fast
    instead of silently running the baseline.
    """
    if spec is None:
        return []
    names = [item.strip() for item in spec.replace(";", ",").split(",") if item.strip()]
    if not names:
        return []
    if any(name == ALL_ALIAS for name in names):
        return list(MECHANISMS.values())
    unknown = sorted(set(names) - set(MECHANISMS) - {OBSERVER.name})
    if unknown:
        raise ValueError(
            f"unknown mechanism(s) {unknown}; choose from {sorted(MECHANISMS)} or 'all'"
        )
    ordered = [MECHANISMS[name] for name in MECHANISMS if name in names]
    if OBSERVER.name in names and not ordered:
        return []  # observer alone is added by compose_* callers
    return ordered


def compose_prompt(
    mechanisms: list[Mechanism],
    *,
    state_dir: str = DEFAULT_STATE_DIR,
    handbook_root: str = DEFAULT_HANDBOOK_ROOT,
) -> str:
    if not mechanisms:
        return ""
    parts = [PROMPT_HEADER.format(state=state_dir)]
    for mech in mechanisms:
        if mech.prompt:
            parts.append(mech.prompt.format(state=state_dir, handbook=handbook_root))
    return "\n\n".join(parts)


_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def compose_bundle_prompt(
    mechanisms: list[Mechanism],
    *,
    state_dir: str = DEFAULT_STATE_DIR,
    handbook_root: str = DEFAULT_HANDBOOK_ROOT,
) -> str:
    """Mechanism prompt plus each skill body inlined as a ``### Skill:`` section."""
    parts = [compose_prompt(mechanisms, state_dir=state_dir, handbook_root=handbook_root)]
    for mech in mechanisms:
        for rel_path, content in mech.skills.items():
            body = content.format(state=state_dir, handbook=handbook_root)
            body = _FRONTMATTER_RE.sub("", body, count=1).strip()
            parts.append(f"### Skill: {rel_path.split('/')[0]}\n{body}")
    return "\n\n".join(part for part in parts if part)


def _hook_command(bin_dir: str, script: str) -> str:
    return f"python3 -I {bin_dir}/{script}"


def build_settings(
    mechanisms: list[Mechanism],
    *,
    state_dir: str = DEFAULT_STATE_DIR,
    include_observer: bool = True,
) -> dict:
    """Return the ``settings.json`` document Claude Code reads for hooks."""
    bin_dir = f"{state_dir}/{BIN_SUBDIR}"
    hooks: dict[str, list[dict]] = {}
    components = list(mechanisms)
    if include_observer and components:
        components.append(OBSERVER)
    for mech in components:
        for spec in mech.hooks:
            hooks.setdefault(spec.event, []).append(
                {
                    "matcher": spec.matcher,
                    "hooks": [
                        {
                            "type": "command",
                            "command": _hook_command(bin_dir, spec.script),
                            "timeout": spec.timeout_sec,
                        }
                    ],
                }
            )
    return {"hooks": hooks} if hooks else {}


def write_bundle(
    mechanisms: list[Mechanism],
    dest: Path,
    *,
    state_dir: str = DEFAULT_STATE_DIR,
    handbook_root: str = DEFAULT_HANDBOOK_ROOT,
) -> Path:
    """Materialize hook scripts, skills, settings and a manifest under ``dest``.

    ``dest`` mirrors the container-side ``state_dir`` layout so it can be
    uploaded as-is: ``bin/`` (scripts), ``skills/`` (SKILL.md files),
    ``settings.json`` and ``manifest.json``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    bin_dir = dest / BIN_SUBDIR
    bin_dir.mkdir(exist_ok=True)
    needed = {"mech_common.py", "policy_locate.py", OBSERVER.hooks[0].script}
    for mech in mechanisms:
        needed.update(spec.script for spec in mech.hooks)
    for script in sorted(needed):
        shutil.copyfile(HOOKS_DIR / script, bin_dir / script)
    skills_dir = dest / SKILLS_SUBDIR
    skills_dir.mkdir(exist_ok=True)
    for mech in mechanisms:
        for rel_path, content in mech.skills.items():
            target = skills_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                content.format(state=state_dir, handbook=handbook_root), encoding="utf-8"
            )
    settings = build_settings(mechanisms, state_dir=state_dir)
    (dest / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    # Harnesses without a skill loader (openai-agents) append this verbatim.
    prompt = compose_bundle_prompt(mechanisms, state_dir=state_dir, handbook_root=handbook_root)
    (dest / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
    manifest = {
        "mechanisms": [m.name for m in mechanisms],
        "state_dir": state_dir,
        "handbook_root": handbook_root,
        "scripts": sorted(needed),
        "skills": sorted(rel for m in mechanisms for rel in m.skills),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dest
