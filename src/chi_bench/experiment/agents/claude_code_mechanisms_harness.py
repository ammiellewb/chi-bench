"""Claude Code harness with toggleable workflow mechanisms for A/B runs.

Behaviour deltas vs the stock Harbor ``ClaudeCode`` harness, all gated on
the ``mechanisms`` agent kwarg (``preflight,policy_trace,criteria_ledger,
consent_gate`` or ``all``; empty = exact baseline):

1. A bundle of stdlib hook scripts, skills, and a ``settings.json`` is built
   on the host from :mod:`chi_bench.experiment.agents.mechanisms` and
   uploaded to ``$CHI_BENCH_MECH_STATE_DIR`` (default ``/logs/agent/mechanisms``)
   so every gate decision and ledger is downloaded with the trial.
2. ``settings.json`` is written to ``$CLAUDE_CONFIG_DIR`` so Claude Code runs
   the PreToolUse/PostToolUse hooks; mechanism skills are copied next to any
   user-supplied ``skills_dir``.
3. The composed mechanism prompt is appended via ``--append-system-prompt``
   (shell-quoted, merged with a user ``append_system_prompt`` kwarg).
4. ``provider_route: litellm`` (or a ``litellm/<id>`` model) points Claude Code
   at a LiteLLM proxy's Anthropic-compatible ``/v1/messages`` endpoint
   (``LITELLM_BASE_URL`` / ``LITELLM_API_KEY``), so open-weight models served
   by vLLM behind LiteLLM run the same mechanisms as Anthropic models.

Example matrix rows::

    - { agent: claude-code-mechanisms, model: anthropic/claude-sonnet-5-5 }
    - { agent: claude-code-mechanisms, model: anthropic/claude-sonnet-5-5,
        agent_kwargs: { mechanisms: preflight } }
"""

from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import EnvVar
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from chi_bench.experiment.agents.mechanisms import (
    DEFAULT_HANDBOOK_ROOT,
    DEFAULT_STATE_DIR,
    Mechanism,
    compose_prompt,
    resolve_mechanisms,
    write_bundle,
)

_CLAUDE_CONFIG_DIR = (EnvironmentPaths.agent_dir / "sessions").as_posix()

STATE_DIR_ENV = "CHI_BENCH_MECH_STATE_DIR"
HANDBOOK_ROOT_ENV = "CHI_BENCH_MECH_HANDBOOK_ROOT"
LITELLM_MODEL_PREFIX = "litellm/"


class ClaudeCodeMechanismsHarness(ClaudeCode):
    """``claude-code`` plus opt-in hook-enforced workflow mechanisms."""

    ENV_VARS = [
        *ClaudeCode.ENV_VARS,
        EnvVar("mechanisms_state_dir", env=STATE_DIR_ENV, default=DEFAULT_STATE_DIR),
        EnvVar("handbook_root", env=HANDBOOK_ROOT_ENV, default=DEFAULT_HANDBOOK_ROOT),
        # Optional overrides of the gated tool sets (comma-separated ns.method).
        EnvVar("preflight_tools", env="CHI_BENCH_MECH_PREFLIGHT_TOOLS"),
        EnvVar("policy_tools", env="CHI_BENCH_MECH_POLICY_TOOLS"),
        EnvVar("ledger_tools", env="CHI_BENCH_MECH_LEDGER_TOOLS"),
    ]

    def __init__(
        self,
        logs_dir: Path,
        mechanisms: str | None = None,
        append_system_prompt: str | None = None,
        provider_route: str | None = None,
        *args,
        **kwargs,
    ):
        self.mechanisms: list[Mechanism] = resolve_mechanisms(mechanisms)
        self._user_system_prompt = append_system_prompt or ""
        self.provider_route = provider_route
        super().__init__(logs_dir, *args, **kwargs)
        self._apply_litellm_route()

    def _apply_litellm_route(self) -> None:
        """Route Claude Code through a LiteLLM proxy when requested.

        Harbor's ``ClaudeCode.run`` reads the base URL from ``os.environ`` and
        the key through ``_get_env`` (``extra_env`` first); with a custom base
        URL it keeps the model id verbatim and pins every model alias to it.
        """
        model = self.model_name or ""
        wants_litellm = self.provider_route == "litellm" or model.startswith(LITELLM_MODEL_PREFIX)
        if not wants_litellm:
            if self.provider_route not in (None, "", "auto"):
                raise ValueError(
                    f"unknown provider_route {self.provider_route!r}; choose 'auto' or 'litellm'"
                )
            return
        base = (self._get_env("LITELLM_BASE_URL") or "").strip()
        if not base:
            raise RuntimeError(
                f"Model {model!r} requires LiteLLM routing, but LITELLM_BASE_URL is not set."
            )
        os.environ["ANTHROPIC_BASE_URL"] = base
        self._extra_env["ANTHROPIC_BASE_URL"] = base
        self._extra_env["ANTHROPIC_API_KEY"] = self._get_env("LITELLM_API_KEY") or "litellm"
        if model.startswith(LITELLM_MODEL_PREFIX):
            self.model_name = model[len(LITELLM_MODEL_PREFIX) :]
            self._init_model_info()

    @staticmethod
    def name() -> str:
        return "claude-code-mechanisms"

    # ── composition helpers ───────────────────────────────────────────

    @property
    def state_dir(self) -> str:
        return self._resolved_env_vars.get(STATE_DIR_ENV, DEFAULT_STATE_DIR)

    @property
    def handbook_root(self) -> str:
        return self._resolved_env_vars.get(HANDBOOK_ROOT_ENV, DEFAULT_HANDBOOK_ROOT)

    @property
    def mechanism_names(self) -> list[str]:
        return [mech.name for mech in self.mechanisms]

    def composed_system_prompt(self) -> str:
        fragment = compose_prompt(
            self.mechanisms, state_dir=self.state_dir, handbook_root=self.handbook_root
        )
        parts = [part for part in (self._user_system_prompt.strip(), fragment) if part]
        return "\n\n".join(parts)

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        prompt = self.composed_system_prompt()
        if prompt:
            extra = f"--append-system-prompt {shlex.quote(prompt)}"
            flags = f"{flags} {extra}".strip()
        return flags

    def _build_register_skills_command(self) -> str | None:
        base = super()._build_register_skills_command()
        if not any(mech.skills for mech in self.mechanisms):
            return base
        mine = (
            f"(cp -r {shlex.quote(self.state_dir + '/skills')}/* "
            "$CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true)"
        )
        return f"{base} && {mine}" if base else mine

    # ── lifecycle ─────────────────────────────────────────────────────

    async def _install_mechanisms(self, environment: BaseEnvironment) -> None:
        state_dir = self.state_dir
        with tempfile.TemporaryDirectory(prefix="chi-mech-") as tmp:
            bundle = write_bundle(
                self.mechanisms,
                Path(tmp) / "bundle",
                state_dir=state_dir,
                handbook_root=self.handbook_root,
            )
            await environment.upload_dir(bundle, state_dir)
        # The agent user must write ledgers next to the uploaded scripts.
        await self.exec_as_root(
            environment,
            command=f"chmod -R a+rwX {shlex.quote(state_dir)}",
            timeout_sec=20,
        )
        settings_src = f"{state_dir}/settings.json"
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(_CLAUDE_CONFIG_DIR)} && "
                f"cp {shlex.quote(settings_src)} {shlex.quote(_CLAUDE_CONFIG_DIR + '/settings.json')}"
            ),
            timeout_sec=10,
        )
        self.logger.info(
            "chi-bench mechanisms installed: %s (state_dir=%s)",
            ",".join(self.mechanism_names) or "<none>",
            state_dir,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self.mechanisms:
            await self._install_mechanisms(environment)
        await super().run(instruction, environment, context)
