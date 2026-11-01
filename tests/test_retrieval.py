from reflexion_agent.retrieval import (
    HashEmbedder,
    ReflectionMemory,
    VectorIndex,
    build_rag_index,
    chunk_text,
)


def test_chunk_text_respects_size_and_keeps_content():
    text = "\n\n".join(f"Title{i}: " + " ".join(["word"] * 120) for i in range(3))
    chunks = chunk_text(text, size=500, overlap=50)
    assert chunks and all(len(c) <= 500 for c in chunks)
    assert all(f"Title{i}" in " ".join(chunks) for i in range(3))


def test_chunk_text_short_text_single_chunk():
    assert chunk_text("hello world", 500, 50) == ["hello world"]
    assert chunk_text("   ", 500, 50) == []


def test_vector_index_returns_nearest_text():
    idx = VectorIndex(HashEmbedder())
    idx.add(["paris is the capital of france", "berlin is the capital of germany", "tokyo japan"])
    assert idx.search("capital of germany", 1) == ["berlin is the capital of germany"]
    assert idx.search("anything", 10) and len(idx.search("anything", 10)) == 3


def test_empty_index_and_memory():
    idx = VectorIndex(HashEmbedder())
    assert idx.search("q", 3) == []
    mem = ReflectionMemory(HashEmbedder(), k=2)
    assert mem.hints("q") == [] and len(mem) == 0
    mem.add("   ")
    assert len(mem) == 0
    mem.add("check the number of territories")
    assert mem.hints("territories") == ["check the number of territories"]


def test_build_rag_index_chunks_all_contexts():
    idx = build_rag_index(["alpha " * 200, "beta " * 10], HashEmbedder(), 500, 50)
    assert len(idx) >= 3


def test_openai_embedder_normalises_and_orders(monkeypatch):
    from types import SimpleNamespace

    from reflexion_agent.retrieval import OpenAIEmbedder

    class FakeEmb:
        def __init__(self):
            self.seen: list[list[str]] = []

        def create(self, model, input):
            self.seen.append(list(input))
            # return out of order to prove sorting by index
            return SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[0.0, 2.0]),
                                         SimpleNamespace(index=0, embedding=[3.0, 0.0])][: len(input)])

    e = OpenAIEmbedder.__new__(OpenAIEmbedder)
    e.model, e.batch, e.query_prefix, e.doc_prefix = "nomic-embed-text", 64, "search_query: ", "search_document: "
    fake = FakeEmb()
    e._client = SimpleNamespace(embeddings=fake)
    v = e.encode(["a", "b"])
    assert v.shape == (2, 2) and abs(v[0, 0] - 1.0) < 1e-6 and abs(v[1, 1] - 1.0) < 1e-6
    assert fake.seen[-1] == ["search_document: a", "search_document: b"]
    e.encode_query(["q"])
    assert fake.seen[-1] == ["search_query: q"]
