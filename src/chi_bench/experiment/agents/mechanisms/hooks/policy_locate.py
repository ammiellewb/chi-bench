"""Policy locator CLI for mechanism 2 (stdlib only).

    python3 -I policy_locate.py "<keywords>" [--book <subdir>] [--max N]

Scores every paragraph under ``<handbook>/references/**/*.md`` by distinct
query-term overlap (exact and prefix matches), prints the best hits as
``file:line | section | excerpt`` and a ready-to-fill policy_trace entry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from . import mech_common as mc
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mech_common as mc  # type: ignore[no-redef]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "are",
    "was",
    "were",
    "has",
    "have",
    "from",
    "into",
    "than",
    "then",
    "when",
    "what",
    "which",
    "who",
    "will",
    "shall",
    "should",
    "must",
    "may",
    "not",
    "any",
    "all",
    "per",
    "its",
    "can",
    "patient",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


def paragraphs(path: Path) -> list[tuple[int, str, str]]:
    """Yield (line_no, section_heading, paragraph_text) for a markdown file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[tuple[int, str, str]] = []
    heading = ""
    buf: list[str] = []
    start = 1
    for index, line in enumerate(lines, start=1):
        if line.startswith("#"):
            if buf:
                out.append((start, heading, " ".join(buf)))
                buf = []
            heading = line.lstrip("#").strip()
            continue
        if not line.strip():
            if buf:
                out.append((start, heading, " ".join(buf)))
                buf = []
            continue
        if not buf:
            start = index
        buf.append(line.strip())
    if buf:
        out.append((start, heading, " ".join(buf)))
    return out


def score(terms: list[str], text: str) -> float:
    body = tokenize(text)
    if not body:
        return 0.0
    body_set = set(body)
    hits = 0.0
    for term in terms:
        if term in body_set:
            hits += 1.0
        elif any(tok.startswith(term) or term.startswith(tok) for tok in body_set):
            hits += 0.5
    if hits == 0:
        return 0.0
    return hits - min(len(body), 400) / 4000.0  # prefer tighter paragraphs


def search(root: Path, query: str, book: str | None, limit: int) -> list[dict[str, object]]:
    terms = tokenize(query)
    base = root / "references"
    if book:
        base = base / book
    if not base.is_dir():
        base = root
    results: list[dict[str, object]] = []
    for path in sorted(base.rglob("*.md")):
        for line_no, heading, text in paragraphs(path):
            value = score(terms, text)
            if value <= 0:
                continue
            results.append(
                {
                    "score": round(value, 3),
                    "file": str(path),
                    "line": line_no,
                    "section": heading,
                    "excerpt": text[:600],
                }
            )
    results.sort(key=lambda r: (-float(r["score"]), str(r["file"]), int(r["line"])))
    return results[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locate handbook clauses by keyword.")
    parser.add_argument("query", help="keywords describing the criterion or rule")
    parser.add_argument("--book", default=None, help="sub-book under references/")
    parser.add_argument("--max", type=int, default=8, dest="limit")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)
    root = mc.handbook_root()
    hits = search(root, args.query, args.book, args.limit)
    mc.log_event("policy_locate", mechanism="policy_trace", query=args.query, hits=len(hits))
    if args.json:
        print(json.dumps(hits, indent=2))
        return 0
    if not hits:
        print(f"No matches for {args.query!r} under {root}. Try fewer or different keywords.")
        return 1
    for hit in hits:
        print(f"{hit['file']}:{hit['line']} | {hit['section']}")
        print(f"    {hit['excerpt']}")
    first = hits[0]
    template = {
        "criterion": "<what is being decided>",
        "source_file": first["file"],
        "section": first["section"],
        "quote": "<copy one sentence from the excerpt verbatim>",
        "interpretation": "<how it applies to this case>",
        "chart_evidence": "<fact from the chart that satisfies or fails it>",
    }
    print("\npolicy_trace.jsonl entry template:")
    print(json.dumps(template))
    return 0


if __name__ == "__main__":
    sys.exit(main())
