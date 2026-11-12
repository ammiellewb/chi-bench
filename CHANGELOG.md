# Changelog

## Unreleased

### Fixes

- `openai-agents` no longer aborts a trial when the model misnames a tool. Separator
  variants (`local_fs_read`, `chart.get_patient`) resolve to the registered tool, and an
  invented tool name returns an error to the model instead of raising
  `ModelBehaviorError`. Exceptions inside a tool (invalid JSON arguments, MCP transport
  errors) also return as tool errors. 33 of the 100 saved Qwen/Gemma trials ended on this
  abort.

### Added

- `openai-agents` writes `agent/stages.jsonl` (per model call and per tool call, flushed as
  they happen) and a `stages` summary plus, on exceptions, `partial_usage` in
  `run_result.json`, so timed-out trials can be diagnosed per stage.
- `openai-agents` accepts the `claude-code-mechanisms` components
  (`agent_kwargs: {mechanisms: ...}`); the same hook scripts run in-process around every
  tool call. Opt-in `progress_nudge_turn` and `turn_budget_warning` nudges.
- `litellm/<id>` model prefix (or `provider_route: litellm`) with `LITELLM_BASE_URL` /
  `LITELLM_API_KEY` for `openai-agents` (any OpenAI-compatible server) and
  `claude-code-mechanisms` (LiteLLM Anthropic passthrough).
- Hook bundles now include `prompt.md`.
- `openai-agents` no longer ends a run on a reply with neither text nor a tool call,
  and asks for `claim_done` (listing the terminal actions observed so far) before
  accepting a text-only final reply. `empty_reply_retries` / `completion_checks`
  kwargs (defaults 1 and 2; 0 disables).
- `policy_trace` and `criteria_ledger` gate only final decision calls; intermediate
  saves (`review.save_md_review`, `review.save_criteria_evaluation`,
  `triage.set_disposition`, `cm_care_plan.save_care_plan`) are no longer gated.
  Simultaneous blocks from several gates arrive as one templated message.
- `payer_letter_center.generate_notification` states in its description that it is
  the only member-notification tool.
- `claim_done` ends the `openai-agents` run once accepted (first call after a terminal
  workflow action, or any second call); a premature first call gets one pushback. A
  Qwen trial had called it 49 times and hit the turn cap.

## 1.1.0 — 2026-10-06

### Fixes

- Restore the published tool-reference path in the Docker image. The directory
  alias `/opt/healthverse-task-assets` points to `/opt/chi-bench-task-assets`, so
  existing task instructions see the selected per-task tool reference.
- Allow a completed payer intake with failed member eligibility to end in the
  existing `denied` state with reason `ELIG_TERMED`. The agent generates and
  delivers a denial letter. Approval and clinical routing are blocked after a
  failed eligibility check.
- Add coverage status and known coverage dates to the existing request-form
  template. The data repair restores pa_t035's termination date from the original
  synthesis facts and replaces its unreachable `requirement_checked` expectation
  with denial and a denial letter. No separate evidence document is added.
- Provide scripts to repair downloaded pa_t035 task copies and regenerate the
  25 saved payer request forms and their marathon copies before rebuilding.
  Remove rendering timestamps from regenerated PDFs so repeat runs preserve
  their hashes when content is unchanged.
- Align the local competition Rules and Overview references with the published
  Kaggle pages.

### Upgrade

After obtaining handbook access and downloading the task data:

```bash
uv sync --extra dev
uv run playwright install chromium
uv run python scripts/repair_pa_t035.py --data-dir data
uv run python scripts/regenerate_payer_request_forms.py --data-dir data
uv run cb data verify
uv run cb docker build
```

On Linux, use `uv run playwright install --with-deps chromium` if browser system
libraries are missing.

This is runtime version **1.1.0**. The downloaded dataset remains pinned to
**chi-bench-v1.0.0**; the scripts apply local corrections. This change does not
publish a new Hugging Face dataset revision. Preserve earlier trial results and
record the runtime commit and image digest for new runs.

### P2P clarification

The P2P workflow is unchanged. The original `MdCaseDecision` records the referral
to P2P; the completed P2P request supplies the outcome, and
`determination.finalize` records the final determination. No second physician
submission is required.

Among the six public payer tasks requiring P2P, t026 and t032 expect the referral
to have `signed_off=false`. Tasks t016, t019, t031, and t036 exclude MD review
from their scored stages. A direct-decision task may require signed physician
review even if an agent chooses P2P instead.
