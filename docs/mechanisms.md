# Harness mechanisms (`claude-code-mechanisms`)

`claude-code-mechanisms` is the stock Harbor `claude-code` harness plus four opt-in,
independently toggleable components. Each component is a **system-prompt fragment**, a set of
**Claude Code tool hooks** (stdlib Python scripts uploaded into the agent container), and
optionally a **skill**. With no mechanisms selected the harness is byte-for-byte the baseline
command, so A/B arms share one harness class and differ only by the listed component.

| Mechanism | Plan idea | What it enforces | Hooks |
|---|---|---|---|
| `preflight` | 1 — executable preflight | Terminal actions (submit, finalize, appeal, respond-to-pend, deliver, complete assessment, finalize care plan, disenroll, P2P summary) are blocked until `preflight.json` names the tool, has `ready: true`, and every check is `pass` with evidence or `na`. After success the preflight is archived and the agent is told to confirm state with a read-only call; after an error it is told to repair input first. | PreToolUse + PostToolUse on chi-bench tools |
| `policy_trace` | 2 — traceable policy use | Decision-bearing calls require `policy_trace.jsonl` entries whose `source_file` resolves inside the handbook and whose `quote` appears verbatim in that file. Ships the `policy-locator` skill and `policy_locate.py` (keyword search over `references/**/*.md`). Logs handbook reads to `policy_reads.jsonl`. | PreToolUse on chi-bench tools; PostToolUse on `Read`/`Bash` |
| `criteria_ledger` | 3 — reasoning before execution | Decision-bearing calls require `criteria_ledger.json`: every criterion has text, a policy source, `met` ∈ {yes, no, unknown}, chart evidence for yes/no, and a next action for unknown; a ledger for another case is rejected. | PreToolUse on chi-bench tools |
| `consent_gate` | 4 — patient autonomy | Not a stop rule (the rubric still requires affirmative enrollment). Ending an outreach call with `consent_obtained: true` or `engagement_status: engaged` requires `consent_record.json` with a verbatim patient quote that appears in the captured outreach turns. Refusal language in a patient reply triggers coaching: acknowledge, do not probe, one low-commitment option. | PreToolUse + PostToolUse on `cm_outreach.*` |

An always-on observer (`observe_hook.py`) is added whenever any mechanism is active. It
writes one `tool_call` event per call and captures outreach turns; it never blocks.

## Running an arm

```yaml
# flat ExperimentConfig or a matrix row
agent: claude-code-mechanisms
model: anthropic/claude-sonnet-5-5
agent_kwargs:
  mechanisms: preflight,policy_trace     # or a single name, or `all`; omit for baseline
```

`configs/experiments/mechanisms_ab_matrix.yaml` holds the six-arm matrix (baseline, one row per
mechanism, all four). Decompose it with `./scripts/run_table.sh mechanisms_ab_matrix`.

Optional kwargs (also settable as `--ae` env on the agent):

| kwarg | env | default |
|---|---|---|
| `mechanisms_state_dir` | `CHI_BENCH_MECH_STATE_DIR` | `/logs/agent/mechanisms` |
| `handbook_root` | `CHI_BENCH_MECH_HANDBOOK_ROOT` | `/workspace/skills/managed-care-operations-handbook` |
| `preflight_tools` | `CHI_BENCH_MECH_PREFLIGHT_TOOLS` | see `preflight_hook.DEFAULT_TERMINAL_TOOLS` |
| `policy_tools` | `CHI_BENCH_MECH_POLICY_TOOLS` | see `policy_trace_hook.DECISION_TOOLS` |
| `ledger_tools` | `CHI_BENCH_MECH_LEDGER_TOOLS` | see `criteria_ledger_hook.DECISION_TOOLS` |

Stock `claude-code` kwargs (`max_turns`, `reasoning_effort`, `append_system_prompt`, `skills_dir`,
…) keep working; a user `append_system_prompt` is placed before the mechanism prompt.

## What lands in the trial

Everything lives under `trials/<name>/agent/mechanisms/`:

- `manifest.json`, `settings.json`, `bin/`, `skills/` — exactly what was installed.
- `events.jsonl` — every `allow`/`deny`/`post_feedback`/`tool_call`/`refusal_detected` event.
- Agent-written records: `preflight.json` → `preflight_history.jsonl`, `postcheck.jsonl`,
  `policy_trace.jsonl`, `policy_reads.jsonl`, `criteria_ledger.json`, `consent_record.json`,
  `outreach_capture.jsonl`.

Summarize arms with:

```bash
uv run python scripts/mechanisms_report.py logs/experiments/mechanisms_ab_matrix
```

Pair it with `scripts/aggregate.py` for pass@k and cost. Denies per trial and tool errors per
trial are the first things to read: a gate that fires often but does not move the reward is
costing turns, and the paper's timeout share (7.6 %) is the budget it eats into.

## How it is wired

- `src/chi_bench/experiment/agents/mechanisms/components.py` — the registry
  (`MECHANISMS`), prompt composition, `settings.json` builder, bundle writer.
- `src/chi_bench/experiment/agents/mechanisms/hooks/` — stdlib-only scripts; each runs as
  `python3 -I <script>` reading the Claude Code hook JSON on stdin. Exit 2 blocks a
  PreToolUse call; PostToolUse feedback uses `decision: block` (prompt) or
  `hookSpecificOutput.additionalContext` (no prompt).
- `src/chi_bench/experiment/agents/claude_code_mechanisms_harness.py` — uploads the bundle,
  writes `$CLAUDE_CONFIG_DIR/settings.json`, copies skills, appends the composed prompt.

Hooks match both native MCP names (`mcp__<server>__ns.method`) and the mcporter CLI form
(`Bash` running `npx mcporter call <alias> ns.method`), so the scripts also work if the
harness is later combined with `tool_mode: cli`.

## Competition note

The gates are domain-generic (named tools and JSON schemas, no task IDs, no task-specific
content), and every record the agent writes stays under `/logs/agent`, outside the judged
workspace. Keep it that way: a mechanism keyed to a specific task would violate the Cup's
circumvention rule.
