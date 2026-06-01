import os
from pathlib import Path
from typing import TypedDict

import requests
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

# Redirect the ONNX model cache into the project directory so it survives Render's
# build snapshot. The default Path.home()/.cache/chroma/... is outside the snapshot.
_ONNX_CACHE = Path(__file__).resolve().parent.parent / "vectorstore" / "onnx_cache"
ONNXMiniLM_L6_V2.DOWNLOAD_PATH = _ONNX_CACHE / ONNXMiniLM_L6_V2.MODEL_NAME

# DefaultEmbeddingFunction.__call__ creates a new ONNXMiniLM_L6_V2() instance on every
# invocation, which re-loads the ONNX model from disk on every request (>120 s on 0.5 CPU).
# Override __call__ to use a single cached instance instead.
_onnx_ef = ONNXMiniLM_L6_V2()
DefaultEmbeddingFunction.__call__ = lambda self, inp: _onnx_ef(inp)  # type: ignore[method-assign]

from app.config import CHROMA_DIR, COLLECTION_NAME
from app.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

_collection: chromadb.Collection | None = None


class Chunk(TypedDict):
    text: str
    document_title: str
    file_name: str
    page: str


class Citation(TypedDict):
    document_title: str
    file_name: str
    page: str


class Snippet(TypedDict):
    document_title: str
    snippet: str


class RAGResult(TypedDict):
    answer: str
    citations: list[Citation]
    snippets: list[Snippet]


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),  # type: ignore[arg-type]
        )
    return _collection


def retrieve_chunks(question: str, top_k: int = 4) -> list[Chunk]:
    collection = get_collection()

    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    documents = (results["documents"] or [[]])[0]
    metadatas = (results["metadatas"] or [[]])[0]

    chunks: list[Chunk] = []
    for doc, meta in zip(documents, metadatas):
        m = meta or {}
        chunks.append({
            "text": doc,
            "document_title": str(m.get("document_title", "")),
            "file_name": str(m.get("file_name", "")),
            "page": str(m.get("page", "")),
        })

    return chunks


def call_openrouter(question: str, context: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def format_citations(chunks: list[Chunk]) -> list[Citation]:
    return [
        {"document_title": c["document_title"], "file_name": c["file_name"], "page": c["page"]}
        for c in chunks
    ]


def format_snippets(chunks: list[Chunk]) -> list[Snippet]:
    return [
        {"document_title": c["document_title"], "snippet": c["text"][:500]}
        for c in chunks
    ]


try:
    get_collection()
    # Force the ONNX model to load (ort.InferenceSession + tokenizer) before the first
    # request arrives. Happens at import time so the gunicorn request timeout doesn't apply.
    _onnx_ef(["warmup"])
except Exception as _startup_err:
    import warnings
    warnings.warn(f"Startup warmup failed: {_startup_err}")


def answer_question(question: str, top_k: int = 4) -> RAGResult:
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "citations": [], "snippets": []}

    chunks = retrieve_chunks(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "I can only answer questions based on the company policy documents provided.",
            "citations": [],
            "snippets": [],
        }

    documents = [c["text"] for c in chunks]
    metadatas = [{"document_title": c["document_title"], "page": c["page"]} for c in chunks]
    context = format_context(documents, metadatas)  # type: ignore[arg-type]
    answer = call_openrouter(question, context)

    return {
        "answer": answer,
        "citations": format_citations(chunks),
        "snippets": format_snippets(chunks),
    }
