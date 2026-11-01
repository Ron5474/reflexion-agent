"""CLI: `reflexion run ...` and `reflexion report`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CONDITIONS, CRITIC_FEEDBACK, STRATEGIES, RunConfig


def _progress(name: str, i: int, n: int) -> None:
    if i == n or i % 10 == 0:
        print(f"  {name}: {i}/{n}", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    from .data import load_hotpotqa
    from .llm import OpenAIChat
    from .retrieval import make_embedder
    from .runner import RunError, run

    cfg = RunConfig.from_env(model=args.model, base_url=args.base_url, n_items=args.n, n_trials=args.trials,
                             seed=args.seed, strategy=args.strategy, critic_feedback=args.critic_feedback,
                             conditions=tuple(args.conditions) if args.conditions else None, out_dir=args.out,
                             embed_backend=args.embed_backend, embed_model=args.embed_model,
                             no_think=True if args.no_think else None)
    print(f"model={cfg.model} base_url={cfg.base_url} n={cfg.n_items} seed={cfg.seed} strategy={cfg.strategy} "
          f"critic={cfg.critic_feedback} conditions={cfg.conditions} embed={cfg.embed_backend}:{cfg.embed_model}", file=sys.stderr)
    records = load_hotpotqa(cfg.n_items, cfg.seed)
    llm = OpenAIChat(cfg.model, cfg.base_url, cfg.api_key, cfg.temperature, cfg.max_tokens, cfg.max_retries,
                     no_think=cfg.no_think)
    embedder = make_embedder(cfg.embed_backend, cfg.embed_model, cfg.base_url, cfg.api_key)
    try:
        out_dir, summary = run(cfg, records, llm, embedder, allow_errors=args.allow_errors, progress=_progress,
                               workers=args.workers)
    except RunError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for cond, v in summary["conditions"].items():
        print(f"{cond:14s} relaxed {v['correct_relaxed']}/{v['n_scored']}  strict {v['correct_strict']}/{v['n_scored']}"
              f"  errors {v['n_errors']}")
    print(f"written to {out_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import load_summaries, markdown_table

    runs = load_summaries(Path(args.results))
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    print(markdown_table(runs))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reflexion")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the HotPotQA evaluation")
    r.add_argument("--model")
    r.add_argument("--base-url")
    r.add_argument("--n", type=int)
    r.add_argument("--trials", type=int)
    r.add_argument("--seed", type=int)
    r.add_argument("--strategy", choices=STRATEGIES)
    r.add_argument("--critic-feedback", choices=CRITIC_FEEDBACK)
    r.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    r.add_argument("--embed-backend", choices=("openai", "st"))
    r.add_argument("--embed-model")
    r.add_argument("--out", default="results")
    r.add_argument("--allow-errors", action="store_true")
    r.add_argument("--no-think", action="store_true", help="append /no_think for Qwen3-style thinking models")
    r.add_argument("--workers", type=int, default=1,
                   help="concurrent questions for full_context/rag_only (actor_critic stays sequential)")
    r.set_defaults(fn=cmd_run)
    rep = sub.add_parser("report", help="print a markdown table of all runs")
    rep.add_argument("--results", default="results")
    rep.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
