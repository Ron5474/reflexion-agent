"""Runs the three conditions and writes auditable results.

results/<model>/<timestamp>/
  config.json        everything that can change a number
  per_question.csv   one row per (condition, question): pred, correct_relaxed, correct_strict, n_attempts, error
  summary.json       per-condition accuracy under both scorers, error counts, call counts
Errors are counted, excluded from accuracy, and make the run fail unless --allow-errors.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .agent import ActorCritic, Outcome
from .config import RunConfig
from .data import Record
from .llm import LLM
from .retrieval import Embedder, ReflectionMemory, VectorIndex, build_rag_index
from .scoring import em_strict


class RunError(RuntimeError):
    pass


def _rag_context(index: VectorIndex, question: str, k: int) -> str:
    return "\n\n".join(index.search(question, k))


def run_condition(
    name: str,
    records: list[Record],
    llm: LLM,
    cfg: RunConfig,
    embedder: Embedder,
    rag: VectorIndex | None,
    progress=None,
    workers: int = 1,
) -> list[Outcome]:
    """actor_critic is always sequential: reflections accumulate across questions in order, and
    parallelising it would change what each question sees. The stateless conditions fan out."""
    memory = ReflectionMemory(embedder, k=cfg.k_memory) if name == "actor_critic" else None
    trials = cfg.n_trials if name == "actor_critic" else 1
    agent = ActorCritic(llm, cfg.strategy, memory, trials, cfg.critic_feedback)
    done = 0
    lock = threading.Lock()

    def one(i_r):
        nonlocal done
        _, r = i_r
        ctx = r.context if name == "full_context" else _rag_context(rag, r.question, cfg.k_retrieve)  # type: ignore[arg-type]
        o = agent.solve(r.qid, r.question, ctx, r.gold)
        if progress:
            with lock:
                done += 1
                progress(name, done, len(records))
        return o

    if memory is not None or workers <= 1:
        return [one(x) for x in enumerate(records)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, enumerate(records)))


def summarize(results: dict[str, list[Outcome]]) -> dict:
    summary: dict = {}
    for name, outs in results.items():
        scored = [o for o in outs if o.correct is not None]
        n = len(outs)
        n_err = n - len(scored)
        relaxed = sum(o.correct for o in scored)  # type: ignore[misc]
        strict = sum(em_strict(o.pred, o.gold) for o in scored)
        summary[name] = {
            "n": n,
            "n_scored": len(scored),
            "n_errors": n_err,
            "correct_relaxed": relaxed,
            "acc_relaxed": round(relaxed / len(scored), 4) if scored else None,
            "correct_strict": strict,
            "acc_strict": round(strict / len(scored), 4) if scored else None,
            "mean_attempts": round(sum(o.n_attempts for o in outs) / n, 3) if n else None,
        }
    return summary


def write_results(out_dir: Path, cfg: RunConfig, results: dict[str, list[Outcome]], extra: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    with (out_dir / "per_question.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "qid", "question", "gold", "pred", "correct_relaxed", "correct_strict",
                    "n_attempts", "error"])
        for name, outs in results.items():
            for o in outs:
                w.writerow([name, o.qid, o.question, o.gold, o.pred,
                            "" if o.correct is None else o.correct,
                            "" if o.correct is None else em_strict(o.pred, o.gold),
                            o.n_attempts, o.error or ""])
    with (out_dir / "attempts.jsonl").open("w") as f:
        for name, outs in results.items():
            for o in outs:
                f.write(json.dumps({"condition": name, "qid": o.qid, "attempts": [asdict(a) for a in o.attempts]}) + "\n")
    summary = {"conditions": summarize(results), **extra}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(cfg: RunConfig, records: list[Record], llm: LLM, embedder: Embedder, allow_errors: bool = False,
        progress=None, workers: int = 1) -> tuple[Path, dict]:
    """Conditions run concurrently with each other (they share nothing but the read-only RAG index);
    within a stateless condition, `workers` questions run at once."""
    t0 = time.time()
    rag = None
    if any(c != "full_context" for c in cfg.conditions):
        rag = build_rag_index([r.context for r in records], embedder, cfg.chunk_size, cfg.chunk_overlap)
    with ThreadPoolExecutor(max_workers=max(1, len(cfg.conditions))) as ex:
        futures = {c: ex.submit(run_condition, c, records, llm, cfg, embedder, rag, progress, workers)
                   for c in cfg.conditions}
        results = {c: f.result() for c, f in futures.items()}
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    safe_model = cfg.model.replace("/", "_").replace(":", "-")
    out_dir = Path(cfg.out_dir) / safe_model / f"{stamp}_{cfg.strategy}_{cfg.critic_feedback}"
    extra = {"model": cfg.model, "strategy": cfg.strategy, "critic_feedback": cfg.critic_feedback,
             "seed": cfg.seed, "n_items": cfg.n_items, "llm_calls": getattr(llm, "calls", None),
             "elapsed_s": round(time.time() - t0, 1), "timestamp": stamp}
    summary = write_results(out_dir, cfg, results, extra)
    n_err = sum(v["n_errors"] for v in summary["conditions"].values())
    if n_err and not allow_errors:
        raise RunError(f"{n_err} LLM errors; results written to {out_dir} but the run is not valid. "
                       f"Fix the endpoint or pass --allow-errors to accept.")
    return out_dir, summary
