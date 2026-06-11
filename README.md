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

## Deployment Notes

For Streamlit Community Cloud, use a lightweight `requirements.txt` such as:

```txt
streamlit>=1.35.0
pypdf>=4.2.0
scikit-learn>=1.4.0
numpy>=1.26.0
python-dotenv>=1.0.1
openai>=1.30.0
reportlab>=4.2.0
```

Do not deploy a full local `pip freeze` file, because it may include packages that only work on your own computer.

---

## Privacy Note

Uploaded papers are processed during the running app session.

The app is not designed as a permanent paper library. Do not upload private or sensitive documents unless you trust the deployment environment.

If you use an OpenAI API key, the retrieved paper passages needed to answer your question may be sent to the OpenAI API.

---

## Limitations

* Offline mode is extractive and may sound less natural than an LLM.
* Search is based on lightweight TF-IDF retrieval.
* Some scanned PDFs may not work because OCR is not included.
* Automatically generated summaries may be imperfect.
* The app does not currently save a personal paper library.

---

## Future Improvements

* Better semantic search with embeddings
* Persistent paper collections
* OCR for scanned PDFs
* More detailed paper comparison tables
* Citation-quality checking
* Better table and figure extraction
* User accounts for private paper libraries
