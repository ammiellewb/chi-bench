# Harness mechanisms (`claude-code-mechanisms`)

`claude-code-mechanisms` is the stock Harbor `claude-code` harness plus four opt-in,
independently toggleable components. Each component is a **system-prompt fragment**, a set of
**Claude Code tool hooks** (stdlib Python scripts uploaded into the agent container), and
optionally a **skill**. With no mechanisms selected the harness is byte-for-byte the baseline
command, so A/B arms share one harness class and differ only by the listed component.
The same mechanisms also run on `openai-agents` for any OpenAI-compatible model; see
[Running on `openai-agents`](#running-on-openai-agents).

| Mechanism | Plan idea | What it enforces | Hooks |
|---|---|---|---|
| `preflight` | 1 — executable preflight | Terminal actions (submit, finalize, appeal, respond-to-pend, deliver, complete assessment, finalize care plan, disenroll, P2P summary) are blocked until `preflight.json` names the tool, has `ready: true`, and every check is `pass` with evidence or `na`. After success the preflight is archived and the agent is told to confirm state with a read-only call; after an error it is told to repair input first. | PreToolUse + PostToolUse on chi-bench tools |
| `policy_trace` | 2 — traceable policy use | Decision-bearing calls require `policy_trace.jsonl` entries whose `source_file` resolves inside the handbook and whose `quote` appears verbatim in that file. Ships the `policy-locator` skill and `policy_locate.py` (keyword search over `references/**/*.md`). Logs handbook reads to `policy_reads.jsonl`. | PreToolUse on chi-bench tools; PostToolUse on `Read`/`Bash` |
| `criteria_ledger` | 3 — reasoning before execution | Decision-bearing calls require `criteria_ledger.json`: every criterion has text, a policy source, `met` ∈ {yes, no, unknown}, chart evidence for yes/no, and a next action for unknown; a ledger for another case is rejected. | PreToolUse on chi-bench tools |
| `consent_gate` | 4 — patient autonomy | Not a stop rule (the rubric still requires affirmative enrollment). Ending an outreach call with `consent_obtained: true` or `engagement_status: engaged` requires `consent_record.json` with a verbatim patient quote that appears in the captured outreach turns. Refusal language in a patient reply triggers coaching: acknowledge, do not probe, one low-commitment option. A care-manager turn with three or more questions, or an overloaded patient reply, triggers pacing coaching: one narrow topic per turn with fresh permission. | PreToolUse + PostToolUse on `cm_outreach.*` |

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

## Running on `openai-agents`

The same four mechanisms run on the `openai-agents` harness for any OpenAI-compatible model
(vLLM, a LiteLLM proxy, OpenRouter, OpenAI direct), so Qwen and Gemma arms can be compared with
the Claude Code arms on identical gates:

```yaml
agent: openai-agents
model: litellm/qwen3-35b            # any id served by LITELLM_BASE_URL; see routing below
agent_kwargs:
  mechanisms: all                   # same names and semantics as claude-code-mechanisms
  progress_nudge_turn: 8            # optional: nudge if no state-changing call by turn 8
  turn_budget_warning: 8            # optional: warn once when 8 turns remain
```

How it maps onto the OpenAI Agents SDK:

- The harness uploads the identical bundle (`bin/`, `skills/`, `settings.json`, `manifest.json`)
  plus a `prompt.md` (mechanism prompt with the skills inlined, since the SDK has no skill
  loader). The runner appends `prompt.md` to the Toolathlon system prompt.
- The runner reads `settings.json` and runs each hook script with `python -I` around every tool
  call, exactly as Claude Code would. A PreToolUse exit 2 returns
  `[blocked by <mechanism>] <reason>` as the tool result without executing the call; PostToolUse
  `decision: block` reasons and `additionalContext` are appended to the tool result.
- Hooks see MCP tools as `mcp__chi_bench__<ns.method>` and the local tools under their Claude
  Code names (`local-fs-read` → `Read` with `file_path`, `local-bash` → `Bash`, …), so the
  policy-read observer and the `policy_locate.py` flow work unchanged.
- Everything lands in the same `trials/<name>/agent/mechanisms/` layout, so
  `scripts/mechanisms_report.py` summarizes both harnesses side by side.

The opt-in nudges target the harness failures seen in the Qwen/Gemma runs (36 PA trials that never
created a case; 3 trials at the 50-turn cap): `progress_nudge_turn` injects one user-role
`[harness notice]` when no tool in `OPENAI_AGENTS_PROGRESS_TOOLS` (default: the case/intake/outreach
entry points and decision calls) has been called by turn N, and `turn_budget_warning` injects one
when N turns remain. Both are off unless set, so an unset row is the exact baseline.

## Harness resilience in `openai-agents` (always on)

`src/chi_bench/experiment/agents/openai_agents_resilience.py` wraps the SDK model and every tool
regardless of mechanisms. It changes nothing the model sees (same prompt, same tool list) unless
the model misbehaves:

| Failure in earlier runs | Behaviour now |
|---|---|
| Model calls `local_fs_read` / `chart.get_patient` for `local-fs-read` / `chart__get_patient`; SDK raises `ModelBehaviorError` and the trial aborts (33 of 100 Qwen/Gemma trials) | Separator variants resolve to the registered tool; the call executes. The trace records `requested_name`. |
| Model invents a tool that does not exist | The call is routed to a hidden sink and the model receives `ERROR: tool '<name>' does not exist … Did you mean …` as the tool result; the run continues. The trace shows the invented name with `unresolved_tool: true`. |
| Invalid JSON arguments or an exception inside a tool (`UserError`, MCP transport error) ends the run | The tool returns `ERROR: tool '<name>' failed: …` to the model. |
| 900 s `AgentTimeoutError` leaves no per-stage data | `agent/stages.jsonl` is appended per model call and per tool call as they happen (latency, tokens, errors, blocks, nudges). `run_result.json` gains `stages` (totals and the five slowest tools) and, on an exception, `partial_usage` for the model calls that completed. |

Read `stages.jsonl` before raising limits: a long `model_sec` with few `tool_calls` is a slow model;
many short model calls with repeated `tool_errors` is a stuck loop.

## LiteLLM and vLLM routing

Both harnesses accept a `litellm/<model-id>` model (or `agent_kwargs: {provider_route: litellm}`
with any id) and read two host variables:

```bash
LITELLM_BASE_URL=http://gpubox:8000/v1   # openai-agents: any OpenAI-compatible root (vLLM works directly)
LITELLM_API_KEY=sk-...                   # optional; "litellm" is sent when unset
```

- `openai-agents` forwards them as `OPENAI_BASE_URL` / `OPENAI_API_KEY`, strips the `litellm/`
  prefix, and defaults to Chat Completions (`api_mode: responses` overrides when the server
  supports it). `cb experiment run` forwards only the LiteLLM pair for this route, not
  `OPENROUTER_API_KEY`.
- `claude-code-mechanisms` sets `ANTHROPIC_BASE_URL` to `LITELLM_BASE_URL`, which must be a
  LiteLLM proxy root exposing the Anthropic-format `/v1/messages` endpoint, and sends
  `LITELLM_API_KEY` as the agent's `ANTHROPIC_API_KEY`. Harbor keeps the model id verbatim and
  pins every Claude Code model alias to it. The native `ANTHROPIC_API_KEY` is untouched for the
  judge.

The routing is unit-tested; neither harness has yet been run live against a LiteLLM proxy.

## How it is wired

- `src/chi_bench/experiment/agents/mechanisms/components.py` — the registry
  (`MECHANISMS`), prompt composition, `settings.json` builder, bundle writer.
- `src/chi_bench/experiment/agents/mechanisms/hooks/` — stdlib-only scripts; each runs as
  `python3 -I <script>` reading the Claude Code hook JSON on stdin. Exit 2 blocks a
  PreToolUse call; PostToolUse feedback uses `decision: block` (prompt) or
  `hookSpecificOutput.additionalContext` (no prompt).
- `src/chi_bench/experiment/agents/claude_code_mechanisms_harness.py` — uploads the bundle,
  writes `$CLAUDE_CONFIG_DIR/settings.json`, copies skills, appends the composed prompt.
- `src/chi_bench/experiment/agents/openai_agents_harness.py` — same upload for `openai-agents`;
  forwards `OPENAI_AGENTS_MECHANISMS` and the nudge settings to the runner.
- `src/chi_bench/experiment/agents/openai_agents_resilience.py` — runner-side model wrapper,
  tool guard, stage log, nudges, and the in-process `MechanismGate`. It is baked into
  `chi-bench:agent` next to the runner and imports no other chi-bench module, so the agent
  image's closed source set stays closed.

Claude Code sanitizes dotted MCP tool names, so hooks see `mcp__healthverse__cm_intake_open_case`; the matcher accepts `.`, `_`, `__` and `-` between namespace and method. Hooks match native MCP names and the mcporter CLI form
(`Bash` running `npx mcporter call <alias> ns.method`), so the scripts also work if the
harness is later combined with `tool_mode: cli`.

## Smoke result (2026-11-08)

One CM task (`cm_asthma_low_coop_001`), Sonnet 5.5, `mechanisms: all`, local Docker: hooks fired
from `$CLAUDE_CONFIG_DIR/settings.json` under Claude Code 2.1.219 in print mode; every gate was
satisfied first time (0 denies, all five record files written); 47 turns, $2.58 agent cost, 5.5 min
agent wall clock; deterministic checks 14/14 and judge rubrics 4/5. The miss was
`judge.cm.outreach.quality`: after consent the agent bundled a screening batch into one turn
(the patient replied "that's a lot at once"), which the rubric treats as a hard fail. That gap
motivated the pacing coaching in `consent_gate`.

## Competition note

The gates are domain-generic (named tools and JSON schemas, no task IDs, no task-specific
content), and every record the agent writes stays under `/logs/agent`, outside the judged
workspace. Keep it that way: a mechanism keyed to a specific task would violate the Cup's
circumvention rule.
