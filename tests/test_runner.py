import json

import pytest

from reflexion_agent.config import RunConfig
from reflexion_agent.data import Record
from reflexion_agent.llm import FakeLLM
from reflexion_agent.report import load_summaries, markdown_table
from reflexion_agent.retrieval import HashEmbedder
from reflexion_agent.runner import RunError, run

RECS = [
    Record("a", "Who wrote Hamlet?", "Shakespeare", "Hamlet: Hamlet is a play by William Shakespeare."),
    Record("b", "Capital of France?", "Paris", "France: Paris is the capital of France."),
]


def _cfg(tmp_path, **kw):
    base = {"model": "fake", "base_url": "http://x", "n_items": 2, "n_trials": 3, "seed": 1, "out_dir": str(tmp_path)}
    base.update(kw)
    return RunConfig(**base)


def test_run_writes_files_and_summary(tmp_path):
    llm = FakeLLM(script={"Hamlet": "Final answer: Shakespeare"}, default="Final answer: Lyon")
    out_dir, summary = run(_cfg(tmp_path), RECS, llm, HashEmbedder())
    for f in ("config.json", "per_question.csv", "summary.json", "attempts.jsonl"):
        assert (out_dir / f).exists()
    c = summary["conditions"]
    assert set(c) == {"full_context", "rag_only", "actor_critic"}
    assert c["full_context"]["correct_relaxed"] == 1 and c["full_context"]["n_errors"] == 0
    assert c["actor_critic"]["mean_attempts"] > 1  # the wrong one retried
    cfg_json = json.loads((out_dir / "config.json").read_text())
    assert cfg_json["api_key"] == "***"


def test_errors_fail_the_run_but_still_write(tmp_path):
    llm = FakeLLM(fail_on="Capital", default="Final answer: Shakespeare")
    with pytest.raises(RunError):
        run(_cfg(tmp_path, conditions=("full_context",)), RECS, llm, HashEmbedder())
    runs = load_summaries(tmp_path)
    assert runs and runs[0][1]["conditions"]["full_context"]["n_errors"] == 1
    assert runs[0][1]["conditions"]["full_context"]["acc_relaxed"] == 1.0  # error excluded, not counted wrong


def test_allow_errors_returns_summary(tmp_path):
    llm = FakeLLM(fail_on="Capital", default="Final answer: Shakespeare")
    _, summary = run(_cfg(tmp_path, conditions=("full_context",)), RECS, llm, HashEmbedder(), allow_errors=True)
    assert summary["conditions"]["full_context"]["n_errors"] == 1


def test_report_table(tmp_path):
    llm = FakeLLM(default="Final answer: Shakespeare")
    run(_cfg(tmp_path, conditions=("full_context",)), RECS, llm, HashEmbedder())
    table = markdown_table(load_summaries(tmp_path))
    assert "| fake | cot | binary | Full context" in table and "1/2 (50%)" in table


def test_config_validation():
    with pytest.raises(ValueError):
        RunConfig(strategy="nope")
    with pytest.raises(ValueError):
        RunConfig(critic_feedback="gold")
    with pytest.raises(ValueError):
        RunConfig(conditions=("rag_only", "bogus"))


def test_config_masks_remote_base_url_but_keeps_local():
    from reflexion_agent.config import RunConfig
    assert RunConfig(base_url="https://provider.example/v1").to_dict()["base_url"] == "***"
    assert RunConfig(base_url="http://localhost:11434/v1").to_dict()["base_url"] == "http://localhost:11434/v1"
