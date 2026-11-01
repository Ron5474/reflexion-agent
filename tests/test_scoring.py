from reflexion_agent.scoring import em_relaxed, em_strict, extract_final_answer, normalize_strict


def test_extract_final_answer_marker():
    assert extract_final_answer("Step 1...\nFinal answer: Paris") == "Paris"
    assert extract_final_answer("final ANSWER: [15 Aug]") == "15 Aug"


def test_extract_final_answer_fallback_last_line():
    assert extract_final_answer("thinking\n\nBerlin\n") == "Berlin"
    assert extract_final_answer("") == ""


def test_em_relaxed_containment_both_ways():
    assert em_relaxed("the flotilla", "flotilla") == 1
    assert em_relaxed("flotilla", "The Flotilla") == 1
    assert em_relaxed("boat", "flotilla") == 0
    assert em_relaxed("", "flotilla") == 0


def test_em_strict_normalises_articles_and_punctuation():
    assert normalize_strict("The Flotilla.") == "flotilla"
    assert em_strict("the Flotilla.", "flotilla") == 1
    assert em_strict("a flotilla of boats", "flotilla") == 0
