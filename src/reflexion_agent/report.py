"""Turn summary.json files into the README results table so numbers trace to a file."""

from __future__ import annotations

import json
from pathlib import Path

LABELS = {"full_context": "Full context (no RAG, no memory)", "rag_only": "RAG only (no memory)",
          "actor_critic": "Actor-Critic (RAG + reflection memory)"}


def load_summaries(results_root: Path) -> list[tuple[Path, dict]]:
    out = []
    for p in sorted(results_root.glob("*/*/summary.json")):
        out.append((p.parent, json.loads(p.read_text())))
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def markdown_table(runs: list[tuple[Path, dict]]) -> str:
    lines = ["| Model | Strategy | Critic feedback | Condition | n | Errors | EM relaxed | EM strict | Run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for path, s in runs:
        for cond, v in s["conditions"].items():
            lines.append(
                f"| {s['model']} | {s['strategy']} | {s['critic_feedback']} | {LABELS.get(cond, cond)} | "
                f"{v['n']} | {v['n_errors']} | {v['correct_relaxed']}/{v['n_scored']} ({_pct(v['acc_relaxed'])}) | "
                f"{v['correct_strict']}/{v['n_scored']} ({_pct(v['acc_strict'])}) | `{path.as_posix()}` |"
            )
    return "\n".join(lines)
