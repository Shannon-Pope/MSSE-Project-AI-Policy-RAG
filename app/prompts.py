from chromadb.api.types import Metadata

SYSTEM_PROMPT = """
You are a company policy assistant.

Rules:
- Answer only using the provided policy context.
- If the answer is not in the context, say:
  "I can only answer questions based on the company policy documents provided."
- Be concise and direct. Answer in 1-3 sentences for simple questions. Stop once the question is answered — do not repeat yourself.
- Cite sources inline using the reference numbers from the context, e.g. [1] or [2].
- Only use square brackets for citation numbers. Never wrap dollar amounts, dates, quantities, or any other values in square brackets.
"""

USER_PROMPT_TEMPLATE = """
User question:
{question}

Policy context:
{context}

Answer with inline citations. Only use citation numbers [1] through [{n_citations}] — these are the only sources available. Do not cite any number outside this range.
"""


def format_context(documents: list[str], metadatas: list[Metadata | None]) -> str:
    """Format retrieved chunks into the numbered context block the prompt expects."""
    parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        title = (meta or {}).get("document_title", "Unknown")
        page = (meta or {}).get("page", "")
        page_str = f" (page {page})" if page else ""
        parts.append(f"[{i}] {title}{page_str}\n{doc}")
    return "\n\n".join(parts)
