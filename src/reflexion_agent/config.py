"""Run configuration. Everything that can change a result lives here and is written to summary.json."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv

STRATEGIES = ("cot", "react", "multihop")
CONDITIONS = ("full_context", "rag_only", "actor_critic")
CRITIC_FEEDBACK = ("binary", "oracle")


@dataclass
class RunConfig:
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "no-key"
    n_items: int = 50
    n_trials: int = 3
    seed: int = 42
    strategy: str = "cot"
    conditions: tuple[str, ...] = CONDITIONS
    # "binary": the critic only learns the answer was wrong (faithful to the Reflexion paper).
    # "oracle": the critic is shown the gold answer (what the Dec 2025 notebook did; leaks labels).
    critic_feedback: str = "binary"
    k_retrieve: int = 3
    k_memory: int = 2
    chunk_size: int = 500
    chunk_overlap: int = 50
    # "openai": POST {base_url}/embeddings with embed_model (Ollama: nomic-embed-text, bge-m3; vLLM: any)
    # "st": sentence-transformers locally (pip install -e ".[st]"), e.g. BAAI/bge-base-en-v1.5
    embed_backend: str = "openai"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.0
    max_tokens: int = 1024
    # Qwen3-style thinking models: append "/no_think" so the budget goes to the answer, not reasoning.
    no_think: bool = False
    max_retries: int = 6
    out_dir: str = "results"
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.conditions = tuple(self.conditions)
        self.validate()

    @classmethod
    def from_env(cls, **overrides) -> RunConfig:
        load_dotenv()
        base = {
            "model": os.getenv("LLM_MODEL", cls.model),
            "base_url": os.getenv("LLM_BASE_URL", cls.base_url),
            "api_key": os.getenv("LLM_API_KEY", cls.api_key),
            "embed_backend": os.getenv("EMBED_BACKEND", cls.embed_backend),
            "embed_model": os.getenv("EMBED_MODEL", cls.embed_model),
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    def validate(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.critic_feedback not in CRITIC_FEEDBACK:
            raise ValueError(f"critic_feedback must be one of {CRITIC_FEEDBACK}")
        bad = [c for c in self.conditions if c not in CONDITIONS]
        if bad:
            raise ValueError(f"unknown conditions {bad}; valid: {CONDITIONS}")
        if self.embed_backend not in ("openai", "st"):
            raise ValueError("embed_backend must be 'openai' or 'st'")
        if self.n_trials < 1 or self.n_items < 1:
            raise ValueError("n_trials and n_items must be >= 1")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["api_key"] = "***" if self.api_key else ""
        # Keep local endpoints for reproducibility; hide which remote provider was used.
        local = any(h in self.base_url for h in ("localhost", "127.0.0.1"))
        d["base_url"] = self.base_url if local else "***"
        return d
