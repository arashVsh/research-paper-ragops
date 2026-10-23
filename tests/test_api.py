from io import BytesIO

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from api.main import app, _requests
from src.pipeline import prepare_paper


def sample_pdf() -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "We propose a neural retrieval method for research papers.")
    pdf.drawString(72, 700, "The method improves document search accuracy in experiments.")
    pdf.save()
    return buffer.getvalue()


def test_api_auth_answers_and_reuses_index(monkeypatch):
    monkeypatch.setenv("PAPERWISE_API_KEYS", "test-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _requests.clear()
    prepare_paper.cache_clear()
    pdf = sample_pdf()
    with TestClient(app) as client:
        data = {"question": "What method is proposed?"}
        files = {"file": ("paper.pdf", pdf, "application/pdf")}
        assert client.post("/ask", data=data, files=files).status_code == 401
        headers = {"X-API-Key": "test-secret"}
        first = client.post("/ask", headers=headers, data=data, files=files)
        assert first.status_code == 200
        body = first.json()
        assert body["used_llm"] is False
        assert body["citations"]
        assert "retrieval" in body["answer"].lower()
        second = client.post("/ask", headers=headers, data=data, files=files)
        assert second.status_code == 200
        assert prepare_paper.cache_info().hits >= 1
        suggestions = client.post("/suggest-questions", headers=headers, files=files)
        assert suggestions.status_code == 200
        assert suggestions.json()["questions"]


def test_api_validation_and_rate_limit(monkeypatch):
    monkeypatch.setenv("PAPERWISE_API_KEYS", "test-secret")
    monkeypatch.setenv("PAPERWISE_RATE_LIMIT", "1")
    _requests.clear()
    headers = {"X-API-Key": "test-secret"}
    with TestClient(app) as client:
        bad = client.post("/suggest-questions", headers=headers,
                          files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
        assert bad.status_code == 422
        limited = client.post("/suggest-questions", headers=headers,
                              files={"file": ("paper.pdf", sample_pdf(), "application/pdf")})
        assert limited.status_code == 429
