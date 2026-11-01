"""Actor and Critic prompts. Text is kept identical to the Dec 2025 notebook so re-runs are comparable."""

from __future__ import annotations

ACTOR_COT_SYSTEM = """You are a helpful QA assistant specialized in multi-hop reasoning.
Think step by step, then provide your final answer.
Always end with: 'Final answer: [your answer]'"""

ACTOR_COT_PROMPT = """You will be given a question and supporting context.
{memory_context}
Question: {question}

Context:
{context}

Think through this step by step, then provide your final answer."""

ACTOR_REACT_SYSTEM = """You are a QA assistant that follows the ReAct framework.
For each question, follow this format:
Thought: [Your reasoning about what information you need]
Observation: [What you find in the context]
Thought: [Your next reasoning step]
Observation: [Additional information]
... (repeat as needed)
Final answer: [your answer]"""

ACTOR_REACT_PROMPT = """You will be given a question and supporting context.
{memory_context}
Question: {question}

Context:
{context}

Use the ReAct format (Thought-Observation cycles) to reason through this, then provide your final answer."""

ACTOR_MULTIHOP_SYSTEM = """You are a QA assistant specialized in multi-hop questions.
For each question:
1. Identify the sub-questions that need to be answered
2. Answer each sub-question using the context
3. Combine the answers to get the final answer
Always end with: 'Final answer: [your answer]'"""

ACTOR_MULTIHOP_PROMPT = """You will be given a multi-hop question and supporting context.
{memory_context}
Question: {question}

Context:
{context}

Break this into sub-questions, answer each, then combine them into your final answer."""

STRATEGIES: dict[str, tuple[str, str]] = {
    "cot": (ACTOR_COT_SYSTEM, ACTOR_COT_PROMPT),
    "react": (ACTOR_REACT_SYSTEM, ACTOR_REACT_PROMPT),
    "multihop": (ACTOR_MULTIHOP_SYSTEM, ACTOR_MULTIHOP_PROMPT),
}

CRITIC_SYSTEM = """You are a meticulous reasoning analyst.
When given a failed answer attempt, you identify what went wrong and provide actionable advice."""

# Notebook behaviour: the critic sees the gold answer. Leaks the label into memory.
CRITIC_PROMPT_ORACLE = """Question: {question}

Context:
{context}

Your previous answer: {prev_answer}
Correct answer: {gold}

Analyze what went wrong. Write a brief, actionable reflection (2-3 sentences) on how to avoid this mistake in the future."""

# Paper-faithful: the critic only learns that the attempt was judged wrong.
CRITIC_PROMPT_BINARY = """Question: {question}

Context:
{context}

Your previous answer: {prev_answer}
This answer was judged incorrect. You are not told the correct answer.

Analyze what likely went wrong in the reasoning or in reading the context. Write a brief, actionable reflection (2-3 sentences) on how to avoid this mistake in the future. Do not guess the answer."""


def actor_prompts(strategy: str) -> tuple[str, str]:
    return STRATEGIES[strategy]


def format_memory(hints: list[str]) -> str:
    if not hints:
        return ""
    return "Past lessons that may help:\n- " + "\n- ".join(hints) + "\n"


def critic_prompt(feedback: str, *, question: str, context: str, prev_answer: str, gold: str) -> str:
    if feedback == "oracle":
        return CRITIC_PROMPT_ORACLE.format(question=question, context=context, prev_answer=prev_answer, gold=gold)
    return CRITIC_PROMPT_BINARY.format(question=question, context=context, prev_answer=prev_answer)
