import hashlib
import re
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.api.types import Metadata
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

_ONNX_CACHE = Path(__file__).resolve().parent.parent / "vectorstore" / "onnx_cache"
ONNXMiniLM_L6_V2.DOWNLOAD_PATH = _ONNX_CACHE / ONNXMiniLM_L6_V2.MODEL_NAME
from pypdf import PdfReader
from bs4 import BeautifulSoup


class DocumentPart(TypedDict):
    text: str
    page: int | None


POLICY_DIR = Path("data/policies")
CHROMA_DIR = "vectorstore/chroma"
COLLECTION_NAME = "company_policies"
BATCH_SIZE = 200


def clean_text(text: str, preserve_newlines: bool = False) -> str:
    text = text.replace("\x00", " ")
    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(lines).strip()
    return re.sub(r"\s+", " ", text).strip()


def load_txt_or_md(file_path: Path) -> list[DocumentPart]:
    text = file_path.read_text(encoding="utf-8")
    return [{"text": text, "page": None}]


def load_html(file_path: Path) -> list[DocumentPart]:
    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return [{"text": text, "page": None}]


def load_pdf(file_path: Path) -> list[DocumentPart]:
    reader = PdfReader(str(file_path))
    pages: list[DocumentPart] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"text": text, "page": page_num})

    return pages


def load_document(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return load_txt_or_md(file_path)
    if suffix in [".html", ".htm"]:
        return load_html(file_path)
    if suffix == ".pdf":
        return load_pdf(file_path)

    raise ValueError(f"Unsupported file type: {file_path}")


def infer_title(file_path: Path) -> str:
    return file_path.stem.replace("_", " ").replace("-", " ")


MIN_CHUNK_LEN = 100


def split_text(text: str, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            boundary = chunk.rfind(" ")
            if boundary > 0:
                end = start + boundary
                chunk = text[start:end]

        if len(chunk.strip()) >= MIN_CHUNK_LEN:
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def main():
    if not POLICY_DIR.exists():
        raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),  # type: ignore[arg-type]
    )

    documents: list[str] = []
    metadatas: list[Metadata] = []
    ids: list[str] = []

    supported_files = (
        list(POLICY_DIR.glob("**/*.md"))
        + list(POLICY_DIR.glob("**/*.txt"))
        + list(POLICY_DIR.glob("**/*.html"))
        + list(POLICY_DIR.glob("**/*.htm"))
        + list(POLICY_DIR.glob("**/*.pdf"))
    )

    if not supported_files:
        raise FileNotFoundError(
            "No policy files found. Add .md, .txt, .html, or .pdf files to data/policies."
        )

    for file_path in supported_files:
        title = infer_title(file_path)
        suffix = file_path.suffix.lower()
        preserve_newlines = suffix in [".txt", ".md"]
        loaded_parts = load_document(file_path)

        for part in loaded_parts:
            cleaned = clean_text(part["text"], preserve_newlines=preserve_newlines)
            chunks = split_text(cleaned)

            for idx, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{file_path.stem}-{part['page'] or 'na'}-{idx}".encode()
                ).hexdigest()

                documents.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "document_title": title,
                    "file_name": file_path.name,
                    "chunk_id": chunk_id,
                    "page": part["page"] or "",
                })

    for i in range(0, len(documents), BATCH_SIZE):
        collection.upsert(
            documents=documents[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE],
        )

    print("Ingestion complete.")
    print(f"Files processed: {len(supported_files)}")
    print(f"Chunks stored: {len(documents)}")
    print(f"Vector DB path: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
