# AI Policy RAG

A Retrieval-Augmented Generation (RAG) application that answers employee questions about company HR policies. Built with Flask, ChromaDB, and OpenRouter.

**Live demo:** https://ai-policy-assistant-7w3z.onrender.com

## How it works

1. Policy documents (`.md`, `.txt`, `.html`, `.pdf`) are chunked and embedded using `all-MiniLM-L6-v2` (ONNX) into a ChromaDB vector store.
2. A user question is embedded with the same model and the top matching chunks are retrieved.
3. The chunks are sent as context to an LLM via OpenRouter, which generates a cited answer.

## Tech stack

- **Flask** — web server
- **ChromaDB** — vector store with ONNX-based embeddings
- **OpenRouter** — LLM API
- **gunicorn** — WSGI server (production)

## Local setup

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=liquid/lfm-2.5-1.2b-thinking:free
```

`OPENROUTER_API_KEY` is required. Get one at [openrouter.ai](https://openrouter.ai).  


### Ingest policy documents

Place documents in `data/policies/`, then run:

```bash
python scripts/ingest.py
```

### Run the app

```bash
python -m app.app
```

Open `http://127.0.0.1:5000` or check health at `http://127.0.0.1:5000/health`.

### Test retrieval

Interactive script to query the vector store directly:

```bash
python scripts/test_retrieval.py
```

## Tests

```bash
pytest
```

Tests cover the retrieval pipeline (`tests/test_retrieval.py`) and Flask endpoints (`tests/test_app.py`).

## Deployment

The app deploys to [Render](https://render.com) via `render.yaml`. The build step runs `scripts/download_onnx.py` (pre-downloads the ONNX embedding model) then `scripts/ingest.py` (populates the vector store); the start command launches gunicorn.

Set `OPENROUTER_API_KEY` as a secret in the Render dashboard (Environment tab). `OPENROUTER_MODEL` is set in `render.yaml`.

For production installs, use `requirements-prod.txt` instead of `requirements.txt` — it contains only the 8 packages the app needs at runtime (no torch, jupyter, or dev tools).
