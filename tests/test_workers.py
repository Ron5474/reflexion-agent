from reflexion_agent.config import RunConfig
from reflexion_agent.data import Record
from reflexion_agent.llm import FakeLLM
from reflexion_agent.retrieval import HashEmbedder
from reflexion_agent.runner import run

RECS = [Record(str(i), f"Question {i}?", f"ans{i}", f"Doc {i}: the answer is ans{i}.") for i in range(12)]


def test_workers_preserve_order_and_results(tmp_path):
    llm = FakeLLM(script={f"Question {i}?": f"Final answer: ans{i}" for i in range(12) if i % 3}, default="Final answer: no")
    cfg = RunConfig(model="fake", base_url="http://x", n_items=12, out_dir=str(tmp_path))
    out_dir, s_par = run(cfg, RECS, llm, HashEmbedder(), workers=4)
    rows_par = (out_dir / "per_question.csv").read_text().splitlines()
    out_dir2, s_seq = run(cfg, RECS, llm, HashEmbedder(), workers=1)
    rows_seq = (out_dir2 / "per_question.csv").read_text().splitlines()
    assert s_par["conditions"] == s_seq["conditions"]
    assert s_par["conditions"]["full_context"]["correct_relaxed"] == 8
    # same question order in the CSV regardless of worker count
    assert [r.split(",")[1] for r in rows_par] == [r.split(",")[1] for r in rows_seq]
