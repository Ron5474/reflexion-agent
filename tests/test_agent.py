from reflexion_agent.agent import ActorCritic
from reflexion_agent.llm import FakeLLM
from reflexion_agent.prompts import CRITIC_SYSTEM
from reflexion_agent.retrieval import HashEmbedder, ReflectionMemory


def _mem():
    return ReflectionMemory(HashEmbedder(), k=2)


def test_correct_first_try_makes_no_critic_call():
    llm = FakeLLM(default="Final answer: flotilla")
    agent = ActorCritic(llm, "cot", _mem(), n_trials=3)
    out = agent.solve("1", "What was he on?", "ctx", "flotilla")
    assert out.correct == 1 and out.n_attempts == 1
    assert all(s != CRITIC_SYSTEM for s, _ in llm.calls)


def test_wrong_answer_triggers_critic_memory_and_retry():
    # First actor call (no hints) is wrong; once a hint is present the actor answers correctly.
    llm = FakeLLM(script={"Past lessons": "Final answer: flotilla"}, default="Final answer: canoe")
    mem = _mem()
    agent = ActorCritic(llm, "cot", mem, n_trials=3, critic_feedback="binary")
    out = agent.solve("1", "What was he on?", "ctx", "flotilla")
    assert out.correct == 1 and out.n_attempts == 2
    assert len(mem) == 1
    critic_calls = [u for s, u in llm.calls if s == CRITIC_SYSTEM]
    assert len(critic_calls) == 1
    assert "flotilla" not in critic_calls[0]  # binary feedback never shows the gold answer


def test_oracle_feedback_shows_gold_to_critic():
    llm = FakeLLM(default="Final answer: canoe")
    agent = ActorCritic(llm, "cot", _mem(), n_trials=2, critic_feedback="oracle")
    agent.solve("1", "q", "ctx", "flotilla")
    critic_calls = [u for s, u in llm.calls if s == CRITIC_SYSTEM]
    assert critic_calls and "Correct answer: flotilla" in critic_calls[0]


def test_trials_are_capped_and_best_attempt_kept():
    llm = FakeLLM(default="Final answer: canoe")
    agent = ActorCritic(llm, "react", _mem(), n_trials=3)
    out = agent.solve("1", "q", "ctx", "flotilla")
    assert out.correct == 0 and out.n_attempts == 3 and out.pred == "canoe"


def test_no_memory_means_single_attempt():
    llm = FakeLLM(default="Final answer: canoe")
    agent = ActorCritic(llm, "multihop", None, n_trials=3)
    out = agent.solve("1", "q", "ctx", "flotilla")
    assert out.n_attempts == 1 and out.correct == 0


def test_llm_error_is_recorded_not_scored():
    llm = FakeLLM(fail_on="Question: q")
    agent = ActorCritic(llm, "cot", _mem(), n_trials=3)
    out = agent.solve("1", "q", "ctx", "flotilla")
    assert out.correct is None and out.error and "simulated" in out.error
