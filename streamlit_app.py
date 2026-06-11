from __future__ import annotations

import time

import streamlit as st
from io import BytesIO
from src.answerer import answer_question
from src.chunking import chunk_pages
from src.config import CONFIG
from src.metrics import compute_query_metrics
from src.mlflow_logger import log_query_event
from src.pdf_loader import load_pdf_from_bytes
from src.question_suggestions import suggest_questions
from src.research_features import (
    build_comparison_rows,
    build_structured_summary,
    reviewer_concern_prompt,
)
from src.retriever import RetrievalIndex

st.set_page_config(
    page_title="Research Paper RAGOps Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Research Paper RAGOps Assistant")
st.caption(
    "Upload research papers, ask questions, get citation-grounded answers, and log query metrics with MLflow."
)

def build_answer_pdf(question: str, answer_text: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from xml.sax.saxutils import escape

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Research Paper RAGOps Assistant", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Question", styles["Heading2"]))
    story.append(Paragraph(escape(question), styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Answer", styles["Heading2"]))

    clean_answer = answer_text.replace("\n", "<br/>")
    story.append(Paragraph(escape(clean_answer), styles["BodyText"]))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=10, value=CONFIG.top_k)
    chunk_size = st.slider("Chunk size", min_value=700, max_value=1800, value=CONFIG.chunk_size, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=400, value=CONFIG.chunk_overlap, step=10)
    enable_mlflow = st.checkbox("Log queries to MLflow", value=True)
    st.divider()
    st.subheader("Optional LLM")
    user_api_key = st.text_input(
        "OpenAI API key",
        value="",
        type="password",
        help="Optional. If empty, the app uses an offline extractive answerer.",
    )
    model_name = st.text_input("OpenAI model", value=CONFIG.openai_model)

    if user_api_key.strip():
        st.success("OpenAI API key detected. LLM mode will be attempted.")
    else:
        st.info("No API key detected. The app will use offline citation-based retrieval.")

uploaded_files = st.file_uploader(
    "Upload one or more research papers as PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > CONFIG.max_uploaded_files:
        st.warning(f"Please upload at most {CONFIG.max_uploaded_files} PDFs at once for the public demo.")
        uploaded_files = uploaded_files[: CONFIG.max_uploaded_files]

    with st.spinner("Extracting text, summarizing papers, and building retrieval index..."):
        all_pages = []
        for uploaded in uploaded_files:
            pdf_bytes = uploaded.read()
            try:
                all_pages.extend(load_pdf_from_bytes(pdf_bytes, uploaded.name))
            except Exception as exc:
                st.error(f"Could not read {uploaded.name}: {type(exc).__name__}: {exc}")

        if not all_pages:
            st.error("No readable text was extracted. The PDFs may be scanned images. OCR is not included in this starter project.")
            st.stop()

        chunks = chunk_pages(all_pages, chunk_size=chunk_size, overlap=chunk_overlap)
        index = RetrievalIndex.build(chunks)
        paper_titles = sorted({p.paper_title for p in all_pages})
        paper_summary = build_structured_summary(chunks)
        dynamic_questions = suggest_questions(chunks)

    st.success(f"Loaded {len(paper_titles)} paper(s), {len(all_pages)} page(s), and {len(chunks)} chunks.")
    with st.expander("Uploaded papers"):
        for title in paper_titles:
            st.write(f"- {title}")

    # ------------------------------------------------------------------
    # Feature 1: structured paper summary cards
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("Structured paper summary")
        st.caption(
            "Automatically extracted from the uploaded paper text. It may be rough; ask a focused question below for a stronger answer."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Problem**")
            st.write(paper_summary.problem)
            st.markdown("**Method**")
            st.write(paper_summary.method)
            st.markdown("**Main contribution**")
            st.write(paper_summary.contribution)
        with c2:
            st.markdown("**Experiments**")
            st.write(paper_summary.experiments)
            st.markdown("**Key result**")
            st.write(paper_summary.key_results)
            st.markdown("**Limitations**")
            st.write(paper_summary.limitations)

        if paper_summary.key_terms:
            st.caption("Key terms: " + ", ".join(paper_summary.key_terms))

    # ------------------------------------------------------------------
    # Session state for chat-like interaction
    # ------------------------------------------------------------------
    def use_suggested_question(selected_question: str) -> None:
        st.session_state["question"] = selected_question
        st.session_state["auto_ask"] = True

    if "question" not in st.session_state:
        st.session_state["question"] = ""
    if "auto_ask" not in st.session_state:
        st.session_state["auto_ask"] = False
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "last_metrics" not in st.session_state:
        st.session_state["last_metrics"] = None
    if "last_warning" not in st.session_state:
        st.session_state["last_warning"] = ""
    if "comparison_rows" not in st.session_state:
        st.session_state["comparison_rows"] = []
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""

    # ------------------------------------------------------------------
    # Feature 2 and 3: reviewer mode + paper comparison mode
    # ------------------------------------------------------------------
    st.subheader("Research tools")
    tool_col1, tool_col2 = st.columns(2)

    with tool_col1:
        if st.button("Reviewer mode: find possible concerns", use_container_width=True):
            st.session_state["question"] = reviewer_concern_prompt()
            st.session_state["auto_ask"] = True
            st.rerun()

    with tool_col2:
        if st.button("Compare uploaded papers", use_container_width=True):
            if len(paper_titles) < 2:
                st.warning("Please upload at least 2 papers to use comparison mode.")
            else:
                st.session_state["question"] = (
                    "Compare the uploaded papers. For each paper, summarize the problem, "
                    "method, key result, and limitation."
                )
                st.session_state["auto_ask"] = True

    if st.session_state["comparison_rows"]:
        with st.expander("Paper comparison table", expanded=True):
            st.dataframe(st.session_state["comparison_rows"], use_container_width=True, hide_index=True)
            st.caption(
                "This comparison is automatically extracted. For a deeper comparison, ask: "
                "'Compare these papers in terms of problem, method, datasets, results, and limitations.'"
            )

    # ------------------------------------------------------------------
    # Answer box: compact and scrollable
    # ------------------------------------------------------------------
    st.subheader("Answer")
    with st.container(height=360, border=True):
        if st.session_state["last_answer"] is None:
            st.info("Choose a suggested question, use reviewer mode, or type your own question below.")
        else:
            if st.session_state["last_warning"]:
                st.warning(st.session_state["last_warning"])

            answer = st.session_state["last_answer"]
            if answer.used_llm:
                st.success("Answer mode: OpenAI API / LLM")
            elif user_api_key.strip():
                st.warning("Answer mode: LLM attempted, but offline retrieval fallback was used")
            else:
                st.info("Answer mode: Offline citation-based retrieval")

            st.markdown(answer.answer)

            if answer.guardrail_warnings:
                st.markdown("**Guardrail warnings**")
                for warning in answer.guardrail_warnings[:5]:
                    st.warning(warning)

        if st.session_state["last_answer"] is not None:
            pdf_bytes = build_answer_pdf(
                question=st.session_state["last_question"],
                answer_text=st.session_state["last_answer"].answer,
            )

            st.download_button(
                label="Export question and answer as PDF",
                data=pdf_bytes,
                file_name="research_paper_answer.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Question box directly below answer
    # ------------------------------------------------------------------
    st.markdown("### Ask a question")
    question_row = st.columns([1, 4])

    with question_row[0]:
        with st.popover("Suggested questions", use_container_width=True):
            for i, q in enumerate(dynamic_questions):
                st.button(
                    q,
                    key=f"suggested_question_{i}",
                    use_container_width=True,
                    on_click=use_suggested_question,
                    args=(q,),
                )

    with question_row[1]:
        question = st.text_area(
            "Your question",
            key="question",
            placeholder="Example: What method does this paper propose, and how does it work in simple terms?",
            height=90,
            label_visibility="collapsed",
        )

    ask = st.button("Ask", type="primary", use_container_width=True)
    should_ask = (ask or st.session_state.get("auto_ask", False)) and question.strip()

    if should_ask:
        st.session_state["auto_ask"] = False

        with st.spinner("Generating answer..."):
            start = time.perf_counter()
            results = index.search(question, top_k=top_k)

            max_score = max((item.score for item in results), default=0.0)
            retrieval_warning = ""
            if max_score < 0.08:
                retrieval_warning = (
                    "The retrieved passages have low relevance, so this answer may be weak. "
                    "Try asking a more specific question if the response is not useful."
                )

            answer = answer_question(
                question=question,
                results=results,
                api_key=user_api_key.strip() or None,
                model=model_name.strip() or None,
            )
            latency = time.perf_counter() - start
            metrics = compute_query_metrics(results, answer, latency)

            if enable_mlflow:
                log_query_event(
                    question=question,
                    paper_titles=paper_titles,
                    params={
                        "top_k": top_k,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "answer_mode": "llm" if answer.used_llm else "offline_extractive",
                    },
                    metrics=metrics,
                    tags={"app": "streamlit", "project": "research-paper-ragops"},
                )

            st.session_state["last_question"] = question
            st.session_state["last_answer"] = answer
            st.session_state["last_results"] = results
            st.session_state["last_metrics"] = metrics
            st.session_state["last_warning"] = retrieval_warning
            st.rerun()

    if st.session_state["last_results"]:
        with st.expander("Retrieved citations and passages"):
            for item in st.session_state["last_results"]:
                c = item.chunk
                st.markdown(
                    f"**[C{item.rank}] {c.paper_title}, pages {c.page_start}-{c.page_end}** "
                    f"— retrieval score: `{item.score:.3f}`"
                )
                st.write(c.text[:1400] + ("..." if len(c.text) > 1400 else ""))
                st.divider()

    if st.session_state["last_metrics"]:
        with st.expander("MLOps / RAGOps metrics logged"):
            st.json(st.session_state["last_metrics"])
else:
    st.info("Upload a PDF to start. For a public demo, try one or two papers first.")

st.divider()
st.caption(
    "Privacy note: this demo keeps processing in the running app session. MLflow logging stores metadata, not full paper text."
)
