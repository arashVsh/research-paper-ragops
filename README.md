# PaperWise AI

PaperWise AI helps you read research papers faster.

Upload one or more PDF papers, ask questions, get citation-grounded answers, compare papers, find possible weaknesses, and export answers as PDF notes.

---

## What You Can Do

* Upload one or more research papers as PDFs
* Ask questions about the uploaded papers
* Get answers with retrieved citations and page references
* Generate an automatic structured summary
* See paper-specific suggested questions
* Compare multiple uploaded papers
* Ask for possible reviewer concerns
* Export the question, answer, and retrieved passages as a PDF
* Use the app with or without an OpenAI API key

---

## How It Works

```text
Upload PDF papers
→ The app extracts text
→ The papers are split into searchable chunks
→ You ask a question
→ The app retrieves relevant passages
→ The app answers using those passages
→ You can inspect the citations and export the answer
```

If you provide an OpenAI API key, PaperWise AI can generate more natural answers. Without an API key, it still works using offline citation-based retrieval.

---

## Main Features

### Paper Summary

After you upload a paper, the app automatically creates a structured summary:

```text
Problem
Method
Main contribution
Experiments
Key result
Limitations
Key terms
```

The summary is automatically extracted and may be rough, but it gives a quick overview of the paper.

---

### Ask Questions

You can ask questions such as:

```text
What is the main contribution of this paper?
What problem does this paper solve?
What method does this paper propose?
How does the method work in simple terms?
What experiments were conducted?
What were the main results?
What are the limitations?
How is this paper different from prior work?
```

---

### Suggested Questions

The app suggests questions based on the uploaded paper content.

Selecting a suggested question automatically sends it and generates an answer.

---

### Reviewer Mode

Reviewer mode helps identify possible concerns, such as:

```text
weak assumptions
missing experiments
missing baselines
unclear evaluation
limited datasets
possible limitations
```

This can help you think more critically about a paper.

---

### Paper Comparison

If you upload at least two papers, you can compare them.

The app compares papers by:

```text
problem
method
key result
limitation
```

If only one paper is uploaded, the app will ask you to upload at least two papers before using comparison mode.

---

### PDF Export

You can export the result as a PDF containing:

```text
question
answer
retrieved citations and passages
```

This is useful for saving literature-review notes.

---

## Using an OpenAI API Key

PaperWise AI works without an API key, but answers are more basic.

To get higher-quality answers, enter your OpenAI API key in the sidebar.

The app uses the key only to generate answers for your current session. Do not share your API key publicly.

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/arashVsh/research-paper-ragops.git
cd research-paper-ragops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

Open the local URL shown in the terminal.

---

## API

Run the API locally with `uvicorn api.main:app --reload`. Interactive documentation is at [http://localhost:8000/docs](http://localhost:8000/docs), and its OpenAPI schema is at `/openapi.json`. Production exposes the same API at `https://YOUR_DOMAIN/api/`, with interactive documentation at `https://YOUR_DOMAIN/api/docs`.

Set `PAPERWISE_API_KEYS` to one or more long random keys separated by commas. Send one in the `X-API-Key` header. Local development permits requests without a key only when no keys are configured. In production (`APP_ENV=production`), the API refuses to start without keys. `/health` and the documentation are public; research endpoints require authentication.

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /ask` | Multipart `file` PDF, `question` (2–2000 characters), optional `top_k` (1–10) | Answer, citations, answer mode, warnings, metrics |
| `POST /suggest-questions` | Multipart `file` PDF | Suggested questions |
| `GET /health` | None | Health status |

```bash
curl -X POST "https://YOUR_DOMAIN/api/ask" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@paper.pdf;type=application/pdf" \
  -F "question=What method does the paper propose?" \
  -F "top_k=5"
```

Each PDF is limited to 20 MB and 250 pages. The API returns 413 for oversized uploads, 422 for invalid or unreadable PDFs, 401 for a missing or invalid key, and 429 when the per-key rate limit is exceeded. The default limit is 30 requests per minute per process; use a shared gateway rate limiter if you scale to multiple replicas. Set `PAPERWISE_CORS_ORIGINS` to a comma-separated list of exact trusted browser origins. Leave it empty for server-to-server clients.

## Production deployment

The included [compose.yaml](compose.yaml) runs the web app, API, and Caddy reverse proxy. Caddy obtains HTTPS certificates for a public domain. On a host with Docker Compose:

1. Point your domain's DNS to the host and open inbound ports 80 and 443.
2. Copy `.env.example` to `.env`. Set `PAPERWISE_DOMAIN`, a random `PAPERWISE_API_KEYS` value, and trusted `PAPERWISE_CORS_ORIGINS` if browser clients need cross-origin access. Optionally set `OPENAI_API_KEY` using the host's secret manager or the private `.env` file.
3. Run `docker compose up -d --build`.
4. Check `https://YOUR_DOMAIN/api/health` and `https://YOUR_DOMAIN/api/docs`.

The container runs as a non-root user. The API uses one worker because its bounded in-memory PDF cache and rate limiter are process-local. The UI keeps a small cached search index so ordinary interactions do not re-extract and re-index uploaded PDFs. OpenAI requests have a timeout and fall back to offline citation-based answers. The UI also works without any OpenAI key. HTTPS requires working DNS and a reachable host; this repository does not include a live deployment or credentials.

For Streamlit Community Cloud, deploy `streamlit_app.py` with `requirements.txt`. That hosts the UI only; use the Compose deployment or another HTTPS ASGI host to expose the API.

---

## Privacy Note

Uploaded papers are processed in memory. A small process-local cache may retain recently processed PDF bytes and search indexes until eviction or restart.

The app is not designed as a permanent paper library. Do not upload private or sensitive documents unless you trust the deployment environment.

If you use an OpenAI API key, the retrieved paper passages needed to answer your question may be sent to the OpenAI API.

---

## Limitations

* Offline mode is extractive and may sound less natural than an LLM.
* Search uses lightweight word and character TF-IDF retrieval; it does not require a hosted embedding service.
* Some scanned PDFs may not work because OCR is not included.
* Automatically generated summaries may be imperfect.
* The app does not currently save a personal paper library.

---

## Future Improvements

* Optional semantic embeddings for large or multilingual paper collections
* Persistent paper collections
* OCR for scanned PDFs
* More detailed paper comparison tables
* Citation-quality checking
* Better table and figure extraction
* User accounts for private paper libraries
