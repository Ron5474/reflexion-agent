"""Actor-Critic loop. The actor answers; if judged wrong, the critic writes a reflection into shared
memory and the actor retries with retrieved reflections, up to n_trials attempts. The best attempt
(by the primary scorer) is kept, ties going to the later attempt, as in the notebook."""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM, LLMError
from .prompts import CRITIC_SYSTEM, actor_prompts, critic_prompt, format_memory
from .retrieval import ReflectionMemory
from .scoring import SCORERS, extract_final_answer


@dataclass
class Attempt:
    trial: int
    raw: str
    pred: str
    correct: int
    hints_used: list[str] = field(default_factory=list)
    reflection: str | None = None


@dataclass
class Outcome:
    qid: str
    question: str
    gold: str
    pred: str
    correct: int | None  # None when an LLM error prevented a verdict
    attempts: list[Attempt]
    error: str | None = None

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)


class ActorCritic:
    def __init__(
        self,
        llm: LLM,
        strategy: str = "cot",
        memory: ReflectionMemory | None = None,
        n_trials: int = 3,
        critic_feedback: str = "binary",
        scorer: str = "relaxed",
    ) -> None:
        self.llm = llm
        self.system, self.template = actor_prompts(strategy)
        self.memory = memory
        self.n_trials = n_trials
        self.critic_feedback = critic_feedback
        self.score = SCORERS[scorer]

    def _act(self, question: str, context: str, hints: list[str]) -> str:
        prompt = self.template.format(memory_context=format_memory(hints), question=question, context=context)
        return self.llm.complete(self.system, prompt)

    def _reflect(self, question: str, context: str, prev: str, gold: str) -> str:
        prompt = critic_prompt(self.critic_feedback, question=question, context=context, prev_answer=prev, gold=gold)
        return self.llm.complete(CRITIC_SYSTEM, prompt)

    def solve(self, qid: str, question: str, context: str, gold: str) -> Outcome:
        attempts: list[Attempt] = []
        try:
            hints = self.memory.hints(question) if self.memory else []
            raw = self._act(question, context, hints)
            pred = extract_final_answer(raw)
            best = Attempt(1, raw, pred, self.score(pred, gold), hints)
            attempts.append(best)
            for trial in range(2, self.n_trials + 1):
                if best.correct == 1 or self.memory is None:
                    break
                reflection = self._reflect(question, context, best.pred, gold)
                self.memory.add(reflection)
                hints = self.memory.hints(question)
                raw = self._act(question, context, hints)
                pred = extract_final_answer(raw)
                att = Attempt(trial, raw, pred, self.score(pred, gold), hints, reflection)
                attempts.append(att)
                if att.correct >= best.correct:
                    best = att
            return Outcome(qid, question, gold, best.pred, best.correct, attempts)
        except LLMError as e:
            return Outcome(qid, question, gold, "", None, attempts, error=str(e))
