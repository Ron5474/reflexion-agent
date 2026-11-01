"""Chunking, embeddings, FAISS index, and the reflection memory store."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

SEPARATORS = ("\n\n", "\n", " ", "")


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Recursive character splitter: split on the coarsest separator that yields pieces <= size,
    merge pieces greedily up to size with `overlap` characters carried between chunks."""

    def _split(t: str, seps: tuple[str, ...]) -> list[str]:
        if len(t) <= size:
            return [t] if t.strip() else []
        sep = seps[0]
        rest = seps[1:]
        parts = t.split(sep) if sep else list(t)
        pieces: list[str] = []
        for p in parts:
            if len(p) > size and rest:
                pieces.extend(_split(p, rest))
            elif p.strip():
                pieces.append(p)
        joiner = sep
        merged: list[str] = []
        cur = ""
        for p in pieces:
            cand = p if not cur else cur + joiner + p
            if len(cand) <= size:
                cur = cand
            else:
                if cur:
                    merged.append(cur)
                tail = cur[-overlap:] if overlap and cur else ""
                cur = (tail + joiner + p) if tail and len(tail + joiner + p) <= size else p
        if cur:
            merged.append(cur)
        return merged

    return _split(text, SEPARATORS)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")


class OpenAIEmbedder:
    """Embeddings from any OpenAI-compatible /v1/embeddings endpoint (Ollama, vLLM, OpenAI)."""

    def __init__(self, model: str, base_url: str, api_key: str = "no-key", batch: int = 64,
                 query_prefix: str | None = None, doc_prefix: str | None = None) -> None:
        from openai import OpenAI

        self.model = model
        self.batch = batch
        # nomic-embed-text is trained with task prefixes; without them retrieval quality drops.
        is_nomic = "nomic" in model.lower()
        self.query_prefix = query_prefix if query_prefix is not None else ("search_query: " if is_nomic else "")
        self.doc_prefix = doc_prefix if doc_prefix is not None else ("search_document: " if is_nomic else "")
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def encode_query(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self._client.embeddings.create(model=self.model, input=texts[i : i + self.batch])
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        vecs = np.asarray(rows, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def make_embedder(backend: str, model: str, base_url: str, api_key: str) -> Embedder:
    if backend == "st":
        return SentenceTransformerEmbedder(model)
    return OpenAIEmbedder(model, base_url, api_key)


class HashEmbedder:
    """Cheap deterministic embedder for tests: bag-of-words hashed into `dim` buckets, L2-normalised."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class VectorIndex:
    """Cosine-similarity index over normalised vectors (FAISS IndexFlatIP)."""

    def __init__(self, embedder: Embedder) -> None:
        import faiss

        self.embedder = embedder
        self.texts: list[str] = []
        self._index = None
        self._faiss = faiss
        self._lock = __import__("threading").Lock()

    def add(self, texts: list[str]) -> None:
        if not texts:
            return
        vecs = self.embedder.encode(texts)
        with self._lock:
            if self._index is None:
                self._index = self._faiss.IndexFlatIP(vecs.shape[1])
            self._index.add(vecs)
            self.texts.extend(texts)

    def search(self, query: str, k: int) -> list[str]:
        if self._index is None or not self.texts:
            return []
        k = min(k, len(self.texts))
        encode_query = getattr(self.embedder, "encode_query", self.embedder.encode)
        q = encode_query([query])
        with self._lock:
            _, ids = self._index.search(q, k)
            return [self.texts[i] for i in ids[0] if i >= 0]

    def __len__(self) -> int:
        return len(self.texts)


class ReflectionMemory:
    """Persistent store of critic reflections. Starts empty, no sentinel entry."""

    def __init__(self, embedder: Embedder, k: int = 2) -> None:
        self.index = VectorIndex(embedder)
        self.k = k

    def add(self, reflection: str) -> None:
        if reflection.strip():
            self.index.add([reflection.strip()])

    def hints(self, question: str) -> list[str]:
        return self.index.search(question, self.k)

    def __len__(self) -> int:
        return len(self.index)


def build_rag_index(contexts: list[str], embedder: Embedder, size: int, overlap: int) -> VectorIndex:
    idx = VectorIndex(embedder)
    chunks: list[str] = []
    for c in contexts:
        chunks.extend(chunk_text(c, size, overlap))
    idx.add(chunks)
    return idx
