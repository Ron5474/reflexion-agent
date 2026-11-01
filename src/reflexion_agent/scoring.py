"""Answer extraction and exact-match scoring.

Two scorers are reported side by side:
- em_relaxed: the notebook's rule (lowercase, strip; correct if equal or either contains the other).
- em_strict: the official HotPotQA normalisation (lowercase, drop punctuation and articles, collapse
  whitespace) with equality only. Numbers on a resume should quote which one they use.
"""

from __future__ import annotations

import re
import string

FINAL_RE = re.compile(r"final answer\s*:", re.IGNORECASE)


def extract_final_answer(response: str) -> str:
    parts = FINAL_RE.split(response)
    if len(parts) > 1:
        return parts[-1].strip().strip("[]").strip()
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else response.strip()


def normalize_strict(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_relaxed(pred: str, gold: str) -> int:
    p, g = pred.strip().lower(), gold.strip().lower()
    if not p or not g:
        return 0
    return int(p == g or g in p or p in g)


def em_strict(pred: str, gold: str) -> int:
    return int(normalize_strict(pred) == normalize_strict(gold))


SCORERS = {"relaxed": em_relaxed, "strict": em_strict}
