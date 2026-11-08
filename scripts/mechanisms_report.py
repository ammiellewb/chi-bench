#!/usr/bin/env python3
"""Summarize hook activity from `claude-code-mechanisms` trials.

Reads every ``<trial>/agent/mechanisms/events.jsonl`` (plus ``result.json``
when present) under one or more roots and prints one row per trial and a
per-root summary: denies/allows per mechanism, refusal coaching events,
policy reads, and the reward when available. Use it next to
``scripts/aggregate.py`` to attribute pass-rate deltas to gate behaviour.

    uv run python scripts/mechanisms_report.py logs/experiments/mechanisms_ab_matrix
    uv run python scripts/mechanisms_report.py <root> --json > report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    except OSError:
        pass
    return records


def _reward(trial_dir: Path) -> float | None:
    result = trial_dir / "result.json"
    if not result.is_file():
        return None
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    verifier = payload.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if isinstance(rewards, dict) and rewards:
        try:
            return float(next(iter(rewards.values())))
        except (TypeError, ValueError):
            return None
    return None


def summarize_trial(trial_dir: Path) -> dict[str, Any] | None:
    mech_dir = trial_dir / "agent" / "mechanisms"
    if not mech_dir.is_dir():
        return None
    manifest: dict[str, Any] = {}
    manifest_path = mech_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    events = _load_jsonl(mech_dir / "events.jsonl")
    denies: Counter[str] = Counter()
    allows: Counter[str] = Counter()
    denied_tools: Counter[str] = Counter()
    tool_calls = 0
    tool_errors = 0
    refusals = 0
    post_feedback = 0
    for event in events:
        kind = event.get("kind")
        mech = str(event.get("mechanism") or "")
        if kind == "deny":
            denies[mech] += 1
            denied_tools[str(event.get("tool"))] += 1
        elif kind == "allow":
            allows[mech] += 1
        elif kind == "tool_call":
            tool_calls += 1
            if event.get("ok") is False:
                tool_errors += 1
        elif kind == "refusal_detected":
            refusals += 1
        elif kind == "post_feedback":
            post_feedback += 1
    return {
        "trial": trial_dir.name,
        "mechanisms": manifest.get("mechanisms", []),
        "reward": _reward(trial_dir),
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "denies": dict(denies),
        "allows": dict(allows),
        "denied_tools": dict(denied_tools.most_common(5)),
        "refusals_detected": refusals,
        "post_feedback": post_feedback,
        "policy_reads": len(_load_jsonl(mech_dir / "policy_reads.jsonl")),
        "policy_citations": len(_load_jsonl(mech_dir / "policy_trace.jsonl")),
        "preflights_consumed": len(_load_jsonl(mech_dir / "preflight_history.jsonl")),
        "ledger_present": (mech_dir / "criteria_ledger.json").is_file(),
        "consent_record_present": (mech_dir / "consent_record.json").is_file(),
    }


def find_trials(root: Path) -> list[Path]:
    return sorted(p.parent.parent for p in root.rglob("agent/mechanisms/events.jsonl"))


def summarize_root(root: Path) -> dict[str, Any]:
    rows = [row for row in (summarize_trial(t) for t in find_trials(root)) if row]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[",".join(row["mechanisms"]) or "baseline"].append(row)
    arms: dict[str, Any] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        rewards = [r["reward"] for r in arm_rows if r["reward"] is not None]
        arms[arm] = {
            "trials": len(arm_rows),
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
            "pass_rate": (sum(1 for r in rewards if r >= 1.0) / len(rewards)) if rewards else None,
            "denies_per_trial": sum(sum(r["denies"].values()) for r in arm_rows) / len(arm_rows),
            "tool_errors_per_trial": sum(r["tool_errors"] for r in arm_rows) / len(arm_rows),
            "policy_reads_per_trial": sum(r["policy_reads"] for r in arm_rows) / len(arm_rows),
            "refusals_per_trial": sum(r["refusals_detected"] for r in arm_rows) / len(arm_rows),
        }
    return {"root": str(root), "trials": rows, "arms": arms}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_text(report: dict[str, Any]) -> None:
    print(f"== {report['root']} ({len(report['trials'])} instrumented trials)")
    header = ("arm", "trials", "pass", "reward", "denies", "errors", "reads", "refusals")
    print("  " + " | ".join(f"{h:>10}" for h in header))
    for arm, stats in report["arms"].items():
        values = (
            arm[:10],
            stats["trials"],
            stats["pass_rate"],
            stats["mean_reward"],
            stats["denies_per_trial"],
            stats["tool_errors_per_trial"],
            stats["policy_reads_per_trial"],
            stats["refusals_per_trial"],
        )
        print("  " + " | ".join(f"{_fmt(v):>10}" for v in values))
    print()
    for row in report["trials"]:
        denies = ", ".join(f"{k}:{v}" for k, v in sorted(row["denies"].items())) or "-"
        print(
            f"  {row['trial'][:48]:<48} reward={_fmt(row['reward']):>5} "
            f"calls={row['tool_calls']:>3} denies=[{denies}] reads={row['policy_reads']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roots", nargs="+", type=Path, help="trial roots to scan")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)
    reports = [summarize_root(root) for root in args.roots]
    if args.json:
        json.dump(reports, sys.stdout, indent=2)
        print()
        return 0
    for report in reports:
        print_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
