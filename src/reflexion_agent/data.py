"""HotPotQA loading and deterministic sampling. Sampling matches the Dec 2025 notebook exactly:
random.seed(seed); shuffle(range(len(ds))); take the first n."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    qid: str
    question: str
    gold: str
    context: str


def build_context(example: dict) -> str:
    """Flatten HotPotQA's context ({title: [...], sentences: [[...], ...]}) into 'Title: sentences' blocks."""
    ctx = example.get("context") or {}
    titles = ctx.get("title") or []
    sentences = ctx.get("sentences") or []
    parts = []
    for i, title in enumerate(titles):
        sents = " ".join(map(str, sentences[i])) if i < len(sentences) else ""
        parts.append(f"{title}: {sents}")
    return "\n\n".join(parts)


def sample_indices(total: int, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    idx = list(range(total))
    rng.shuffle(idx)
    return idx[:n]


def load_hotpotqa(n: int, seed: int, split: str = "validation") -> list[Record]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    chosen = sample_indices(len(ds), n, seed)
    subset = ds.select(chosen)
    return [
        Record(qid=str(ex.get("id", i)), question=ex["question"], gold=ex["answer"], context=build_context(ex))
        for i, ex in zip(chosen, subset)
    ]
