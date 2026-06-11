# Research Paper RAGOps Assistant

A production-style research paper assistant for people who upload papers daily and ask questions about them.

The app lets users upload one or more PDFs, automatically extracts text, chunks the papers, retrieves relevant passages, answers questions with citations, suggests useful questions, and logs query/quality metadata with MLflow.

It is designed as a portfolio-ready **LLMOps / RAGOps / MLOps** project rather than a notebook demo.

## Features

- PDF upload and text extraction
- Paper chunking with page-aware citations
- Retrieval over uploaded papers using TF-IDF
- Optional LLM answer generation with OpenAI
- Offline fallback answer generation if no API key is provided
- Suggested research questions for each uploaded paper set
- Prompt-injection and unsafe-instruction detection
- MLflow logging for query events, retrieval metrics, latency, and guardrail flags
- Streamlit public UI
- FastAPI backend example
- Docker support
- GitHub Actions CI
- Unit tests

## Why this solves a real problem

Many researchers repeatedly upload papers to ChatGPT and ask questions such as:

- What is the main contribution?
- What problem does this paper solve?
- What are the limitations?
- What are the assumptions?
- How is this different from previous work?
- What experiments should I reproduce?
- What could reviewers criticize?

This project turns that workflow into a reusable web app.

## Quick start

### 1. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the local URL shown in the terminal.

### 4. Optional: use OpenAI for higher-quality answers

Create a `.env` file:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

The app still works without an API key, but it uses an extractive fallback answerer.

## Run MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

You will see logged query runs with metrics such as retrieval score, latency, number of chunks, answer length, and guardrail flags.

## Run FastAPI backend

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker

```bash
docker build -t research-paper-ragops .
docker run -p 8501:8501 --env-file .env research-paper-ragops
```

Open:

```text
http://localhost:8501
```

## Deploy publicly

The easiest public deployment options are:

### Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Set the main file to `streamlit_app.py`.
5. Add `OPENAI_API_KEY` as a secret if you want LLM answers.

### Hugging Face Spaces

1. Create a new Space.
2. Choose Streamlit.
3. Upload the repository files.
4. Add `OPENAI_API_KEY` as a secret if needed.

If you want everyone to use the app without entering their own key, you need to provide a backend API key yourself and control rate limits/cost.

## Project structure

```text
research-paper-ragops/
├── streamlit_app.py
├── api/
│   └── main.py
├── src/
│   ├── answerer.py
│   ├── chunking.py
│   ├── config.py
│   ├── guardrails.py
│   ├── metrics.py
│   ├── mlflow_logger.py
│   ├── pdf_loader.py
│   ├── question_suggestions.py
│   ├── retriever.py
│   └── schemas.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Resume bullet

Built a public Research Paper RAGOps Assistant using Streamlit, FastAPI, MLflow, Docker, and automated tests; implemented PDF ingestion, retrieval-augmented question answering, citation-grounded responses, suggested research questions, prompt-injection checks, query observability, and MLflow logging for retrieval quality, latency, and guardrail metrics.

## Limitations

- The offline answerer is extractive and not as fluent as an LLM.
- TF-IDF retrieval is lightweight and easy to deploy, but dense embeddings can improve semantic search.
- Public deployment with an LLM requires cost control, authentication, or per-user API keys.

## Future improvements

- Add Chroma/Qdrant vector database
- Add dense embeddings
- Add multi-user authentication
- Add persistent document collections
- Add paper comparison mode
- Add reviewer-style critique mode
- Add automatic paper cards and model cards
- Add monitoring dashboard
