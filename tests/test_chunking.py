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


def test_overlap_keeps_original_page_citations():
    pages = [
        PaperPage("p", "Paper", 1, "alpha " * 35),
        PaperPage("p", "Paper", 2, "beta " * 35),
    ]
    chunks = chunk_pages(pages, chunk_size=170, overlap=40)
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)
    assert chunks[-1].page_end == 2
