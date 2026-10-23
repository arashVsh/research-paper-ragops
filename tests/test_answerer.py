from src import answerer
from src.schemas import PaperChunk, RetrievedChunk


def test_online_failure_falls_back_to_cited_offline_answer(monkeypatch):
    chunk = PaperChunk("c", "p", "Paper", 2, 2,
                       "We propose a retrieval model that improves the accuracy of document search.")
    results = [RetrievedChunk(chunk, 0.8, 1)]

    def fail(*args, **kwargs):
        raise TimeoutError("upstream unavailable")

    monkeypatch.setattr(answerer, "_llm_answer", fail)
    result = answerer.answer_question("What model is proposed?", results, api_key="test")
    assert result.used_llm is False
    assert "[C1]" in result.answer
    assert result.citations
