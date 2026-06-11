from src.retriever import RetrievalIndex
from src.schemas import PaperChunk


def test_retriever_finds_relevant_chunk():
    chunks = [
        PaperChunk("1", "p1", "Paper", 1, 1, "Adversarial examples attack neural networks."),
        PaperChunk("2", "p1", "Paper", 2, 2, "Parquet files store columnar data."),
    ]
    index = RetrievalIndex.build(chunks)
    results = index.search("neural network attack", top_k=1)
    assert len(results) == 1
    assert "Adversarial" in results[0].chunk.text
