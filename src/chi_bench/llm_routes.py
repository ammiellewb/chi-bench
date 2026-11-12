"""Model routes for the trusted-side LLM roles: judge and simulators.

chi-Bench calls a model from three trusted places: the workspace judge, the
P2P counterpart simulator and the care-management patient simulator. By
default all three use Anthropic (the paper protocol: Opus 4.7 judge, Sonnet
simulators). Setting a role's provider to ``openai`` sends it to any
OpenAI-compatible Chat Completions server instead (vLLM, a LiteLLM proxy,
OpenAI, Azure), so a run needs no Anthropic key at all.

Environment (all optional; unset = Anthropic as before):

``CHI_BENCH_JUDGE_PROVIDER``   ``anthropic`` (default) or ``openai``
``CHI_BENCH_JUDGE_BASE_URL``   OpenAI-compatible root, e.g. ``http://gpubox:8000/v1``
``CHI_BENCH_JUDGE_API_KEY``    sent as a bearer token (``EMPTY`` when unset)
``CHI_BENCH_JUDGE_MODEL``      required for ``openai``
``CHI_BENCH_JUDGE_EXTRA_BODY`` JSON merged into every request

``CHI_BENCH_SIM_PROVIDER`` / ``_BASE_URL`` / ``_API_KEY`` / ``_MODEL`` /
``_EXTRA_BODY`` / ``_MAX_TOKENS`` do the same for both simulators. When
``CHI_BENCH_SIM_PROVIDER`` is unset the simulators follow the judge route, so
one block of judge settings moves every trusted model call. Role-specific
models still win: ``CHI_BENCH_P2P_SIMULATOR_MODEL`` and
``CHI_BENCH_PATIENT_SIM_MODEL`` (Anthropic aliases such as ``sonnet`` are
ignored on the ``openai`` route).

Using a non-Anthropic judge deviates from the paper protocol; scores are not
directly comparable with Opus-judged runs. Provenance records the route.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

PROVIDERS = ("anthropic", "openai")
DEFAULT_JUDGE_MODEL = "claude-opus-4-7"
DEFAULT_SIM_MAX_TOKENS = 4096
_ANTHROPIC_ALIASES = {"sonnet", "opus", "haiku"}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMRouteError(RuntimeError):
    """Misconfigured route or a failed OpenAI-compatible request."""


@dataclass(frozen=True)
class LLMRoute:
    role: str
    provider: str
    model: str | None
    base_url: str | None = None
    api_key: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    def describe(self) -> dict[str, Any]:
        """Provenance-safe description (never includes the key)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url if self.is_openai else None,
        }


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _provider(name: str, default: str) -> str:
    value = (_env(name) or default).lower()
    if value not in PROVIDERS:
        raise LLMRouteError(f"{name}={value!r}; choose one of {', '.join(PROVIDERS)}")
    return value


def _extra_body(name: str) -> dict[str, Any]:
    raw = _env(name)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMRouteError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMRouteError(f"{name} must be a JSON object")
    return parsed


def _require_openai_settings(
    role: str, base_url: str | None, model: str | None, prefix: str
) -> None:
    if not base_url:
        raise LLMRouteError(f"{role} uses the openai provider but {prefix}_BASE_URL is not set")
    if not model:
        raise LLMRouteError(f"{role} uses the openai provider but {prefix}_MODEL is not set")


def judge_route() -> LLMRoute:
    provider = _provider("CHI_BENCH_JUDGE_PROVIDER", "anthropic")
    if provider == "anthropic":
        return LLMRoute("judge", provider, _env("CHI_BENCH_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL)
    base_url = _env("CHI_BENCH_JUDGE_BASE_URL")
    model = _env("CHI_BENCH_JUDGE_MODEL")
    _require_openai_settings("The judge", base_url, model, "CHI_BENCH_JUDGE")
    return LLMRoute(
        "judge",
        provider,
        model,
        base_url=base_url,
        api_key=_env("CHI_BENCH_JUDGE_API_KEY"),
        extra_body=_extra_body("CHI_BENCH_JUDGE_EXTRA_BODY"),
    )


def simulator_route(role: str, model_env: str) -> LLMRoute:
    """Route for a simulator role; ``model_env`` names its role-specific model variable."""
    role_model = _env(model_env)
    sim_provider = _env("CHI_BENCH_SIM_PROVIDER")
    if sim_provider is None:
        judge = judge_route()
        if not judge.is_openai:
            return LLMRoute(role, "anthropic", role_model)
        model = role_model if role_model and role_model not in _ANTHROPIC_ALIASES else None
        return LLMRoute(
            role,
            "openai",
            model or _env("CHI_BENCH_SIM_MODEL") or judge.model,
            base_url=_env("CHI_BENCH_SIM_BASE_URL") or judge.base_url,
            api_key=_env("CHI_BENCH_SIM_API_KEY") or judge.api_key,
            extra_body=_extra_body("CHI_BENCH_SIM_EXTRA_BODY") or judge.extra_body,
            max_tokens=_sim_max_tokens(),
        )
    provider = _provider("CHI_BENCH_SIM_PROVIDER", "anthropic")
    if provider == "anthropic":
        return LLMRoute(role, provider, role_model)
    model = role_model if role_model and role_model not in _ANTHROPIC_ALIASES else None
    model = model or _env("CHI_BENCH_SIM_MODEL")
    base_url = _env("CHI_BENCH_SIM_BASE_URL")
    _require_openai_settings(f"The {role}", base_url, model, "CHI_BENCH_SIM")
    return LLMRoute(
        role,
        provider,
        model,
        base_url=base_url,
        api_key=_env("CHI_BENCH_SIM_API_KEY"),
        extra_body=_extra_body("CHI_BENCH_SIM_EXTRA_BODY"),
        max_tokens=_sim_max_tokens(),
    )


def _sim_max_tokens() -> int:
    raw = _env("CHI_BENCH_SIM_MAX_TOKENS")
    try:
        return int(raw) if raw else DEFAULT_SIM_MAX_TOKENS
    except ValueError as exc:
        raise LLMRouteError(f"CHI_BENCH_SIM_MAX_TOKENS={raw!r} is not an integer") from exc


def strip_reasoning(text: str) -> str:
    """Drop inline ``<think>`` blocks some servers leave in the content."""
    return _THINK_RE.sub("", text or "").strip()


def chat_completion_raw(
    route: LLMRoute,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    timeout_s: float = 300.0,
    attempts: int = 3,
) -> dict[str, Any]:
    """POST ``/chat/completions``; retry transport errors, 429 and 5xx."""
    if not route.is_openai or not route.base_url:
        raise LLMRouteError(f"{route.role} route is not an OpenAI-compatible route")
    body: dict[str, Any] = {
        "model": route.model,
        "messages": messages,
        "max_tokens": max_tokens,
        **route.extra_body,
    }
    if tools:
        body["tools"] = tools
        body.setdefault("tool_choice", "auto")
    headers = {"Authorization": f"Bearer {route.api_key or 'EMPTY'}"}
    url = route.base_url.rstrip("/") + "/chat/completions"
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, json=body, headers=headers, timeout=timeout_s)
        except requests.RequestException as exc:
            last = exc
        else:
            if response.status_code == 200:
                return response.json()
            detail = response.text[:500]
            last = LLMRouteError(
                f"{route.role} request failed: HTTP {response.status_code}: {detail}"
            )
            if response.status_code not in (408, 409, 425, 429) and response.status_code < 500:
                raise last
        if attempt < attempts:
            time.sleep(2.0 * attempt)
    raise LLMRouteError(f"{route.role} request failed after {attempts} attempts: {last}")


def chat_text(
    route: LLMRoute,
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    timeout_s: float = 300.0,
) -> tuple[str, int]:
    """One text turn: returns ``(reply, total_tokens)``."""
    payload = chat_completion_raw(
        route,
        [{"role": "system", "content": system}, *messages],
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    choice = (payload.get("choices") or [{}])[0]
    text = strip_reasoning(str((choice.get("message") or {}).get("content") or ""))
    if not text:
        raise LLMRouteError(
            f"{route.role} returned no text (finish_reason={choice.get('finish_reason')!r}). "
            "If the model spends the budget thinking, raise CHI_BENCH_SIM_MAX_TOKENS or set "
            'CHI_BENCH_SIM_EXTRA_BODY=\'{"chat_template_kwargs": {"enable_thinking": false}}\'.'
        )
    usage = payload.get("usage") or {}
    total = usage.get("total_tokens")
    if not isinstance(total, int):
        total = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    return text, total
