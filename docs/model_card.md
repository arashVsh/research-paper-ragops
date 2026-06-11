# Model / System Card: Research Paper RAGOps Assistant

## Purpose

This system helps researchers ask questions about uploaded research papers and receive citation-grounded answers.

## Intended users

- Graduate students
- Researchers
- Engineers reading technical papers
- Research labs and reading groups

## Intended use

- Summarizing papers
- Understanding methods and experiments
- Finding limitations
- Generating follow-up questions
- Comparing retrieved evidence from documents

## Out-of-scope use

- Treating the generated answer as a verified peer-review decision
- Legal, medical, or financial advice
- Answering questions without supporting paper evidence

## Data

The user uploads PDFs. Text is extracted and chunked at runtime.

## Retrieval

The starter implementation uses TF-IDF retrieval for simplicity and deployability. Dense vector search can be added later.

## Generation

The app supports two modes:

1. Offline extractive answerer
2. Optional OpenAI LLM answerer if an API key is supplied

## Safety and security

- Uploaded paper text is treated as untrusted data.
- The app detects common prompt-injection patterns.
- MLflow logging stores metadata and hashes, not full paper text.

## Limitations

- Scanned PDFs require OCR, which is not included by default.
- Offline mode is less fluent than LLM mode.
- TF-IDF retrieval may miss semantic matches.
- Public LLM use requires rate limiting and cost management.
