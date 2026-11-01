from reflexion_agent.llm import strip_think


def test_strip_think_removes_closed_blocks():
    assert strip_think("<think>\nreasoning\n</think>\nFinal answer: Paris") == "Final answer: Paris"
    assert strip_think("no tags here") == "no tags here"


def test_strip_think_unterminated_is_empty():
    assert strip_think("<think> still thinking and cut off") == ""
