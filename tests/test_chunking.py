from src.chunking import chunk_pages
from src.schemas import PaperPage


def test_chunk_pages_creates_chunks():
    pages = [
        PaperPage(
            paper_id="abc",
            paper_title="Test Paper",
            page_number=1,
            text="This is a test page about adversarial machine learning. " * 30,
        )
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=40)
    assert len(chunks) >= 1
    assert chunks[0].paper_title == "Test Paper"
    assert chunks[0].page_start == 1
