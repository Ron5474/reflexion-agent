from reflexion_agent.data import build_context, sample_indices


def test_build_context_formats_title_and_sentences():
    ex = {"context": {"title": ["A", "B"], "sentences": [["s1.", "s2."], ["t1."]]}}
    assert build_context(ex) == "A: s1. s2.\n\nB: t1."


def test_build_context_handles_missing():
    assert build_context({}) == ""
    assert build_context({"context": {"title": ["A"], "sentences": []}}) == "A: "


def test_sample_indices_is_deterministic_and_matches_notebook_rule():
    import random

    a = sample_indices(1000, 50, 42)
    b = sample_indices(1000, 50, 42)
    assert a == b and len(a) == 50 and len(set(a)) == 50
    random.seed(42)
    idx = list(range(1000))
    random.shuffle(idx)
    assert a == idx[:50]
