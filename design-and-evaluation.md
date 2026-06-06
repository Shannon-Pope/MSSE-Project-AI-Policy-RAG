# Design and Evaluation

## Architecture Overview

The application is a standard RAG pipeline: documents are chunked and embedded into a local vector store at ingest time; at query time the user's question is embedded, the top-k most similar chunks are retrieved, and those chunks are injected into an LLM prompt that generates a cited answer. A Flask web server exposes the UI and API.

## Design Decisions

### Embedding Model — `all-MiniLM-L6-v2` (ONNX)

ChromaDB's built-in `ONNXMiniLM_L6_V2` was chosen as the embedding model. It runs the `all-MiniLM-L6-v2` sentence transformer locally via ONNX Runtime with no external API calls or cost. The model is well-established for semantic similarity tasks and produces 384-dimensional embeddings that are compact enough for a small corpus. Running the model locally via ONNX also eliminates network latency and API rate limits at query time, and allows the same model to be used consistently at both ingest and retrieval time without configuration drift.

### Chunking Strategy — 900 characters, 150-character overlap

Documents are split into chunks of approximately 900 characters with a 150-character overlap. This size was chosen to fit meaningfully within `all-MiniLM-L6-v2`'s 256-token input window while keeping each chunk focused enough to embed a single policy concept. A larger chunk risks diluting the embedding signal across unrelated content; a smaller chunk risks splitting a single policy rule across two chunks and losing context. The 150-character overlap prevents a relevant sentence being cut at a boundary and missed by retrieval.

### Top-k Retrieval — k = 4

Four chunks are retrieved per query. This provides enough context to cover multi-step policy questions (e.g. a procedure with several steps across different paragraphs) without exceeding the LLM's useful context window or inflating prompt size unnecessarily. With a small corpus of 9 documents, k=4 retrieves from multiple source documents when relevant, supporting cross-policy answers.

### Vector Store — ChromaDB (local persistent)

ChromaDB was selected as the vector store because it is free, requires no external service, and persists the index to disk so it survives server restarts and Render deployments. It integrates directly with the ONNX embedding function used at ingest, making the retrieval path consistent. For a corpus of this size (9 documents, hundreds of chunks) a local store is sufficient; a hosted store like Pinecone would add cost and a network dependency without meaningful benefit.

### LLM and API — OpenRouter (free tier)

The LLM is accessed via OpenRouter's free tier API using the `liquid/lfm-2.5-1.2b-thinking:free` model. OpenRouter was chosen because it provides access to capable LLMs at zero cost, which satisfies the project requirement to use free-tier options. The API is called directly over HTTP using `httpx` rather than through a framework, keeping the dependency surface small. The trade-off of the free tier is high latency (p50 ≈ 26s), which is a consequence of shared free-tier infrastructure rather than application design.

### Prompt Design

The prompt uses a two-part structure: a system prompt that sets the assistant role and rules, and a user prompt that injects the numbered context chunks alongside the question. Chunks are prefixed with reference numbers ([1], [2], etc.) so the LLM can produce inline citations that trace directly back to source documents and pages. The system prompt enforces three guardrails: refusing to answer outside the corpus, limiting answer length (target under 200 words for simple questions, up to 400 for multi-step procedures), and always citing sources. This keeps answers grounded and auditable.

### Web Framework — Flask (no LangChain)

Flask was chosen for its simplicity — it requires minimal boilerplate for a single-endpoint JSON API backed by a Python function. The RAG pipeline was implemented manually rather than via LangChain, which gave full visibility into the retrieval and prompt-construction logic and avoided pulling in a large dependency for what amounts to a fetch call and string formatting.

## Evaluation Approach

I evaluated my RAG application using 18 test questions across PTO, travel, expenses, remote work, and security topics, including out-of-scope topics.

The evaluation measured:

- Groundedness: whether the answer was fully supported by retrieved policy evidence
- Citation Accuracy: whether the cited source matched the answer
- Latency: request-to-answer time in seconds (s)

## Evaluation Results

| Metric | Result |
|---|---:|
| Number of Questions | 18 |
| Groundedness | 86.1% |
| Citation Accuracy | 61.1% |
| Latency p50 | 25.97 s |
| Latency p95 | 33.19 s |

## Scoring Method

Groundedness and citation accuracy were manually scored using:

- 1.0 = correct
- 0.5 = partially correct
- 0.0 = incorrect

Latency was measured automatically using Python's `time.perf_counter()` around each call to the RAG pipeline.

## Observations

The system performed best on direct policy questions where the answer was explicitly stated in a single source document. Groundedness was strong (86.1%) because the system prompt strictly instructs the model to answer only from the provided context.

Citation accuracy (61.1%) was weaker. The LLM produces inline citations by number (e.g. [1], [2]) referring to the numbered context blocks injected in the prompt. In cases where the model cited a number that did not correspond to the specific passage supporting its answer — or omitted a citation entirely — the response was scored as partially or fully incorrect on this metric. This is a known limitation of inline-citation prompting with small free-tier models.

Latency (p50 = 25.97s, p95 = 33.19s) is high. This is entirely attributable to the free-tier LLM on OpenRouter, which queues requests on shared infrastructure. Embedding and ChromaDB retrieval are fast (typically under 1s combined). A paid model API would reduce end-to-end latency significantly.

## Known Limitations

- Evaluation set is small (18 questions)
- Groundedness and citation accuracy were manually scored, introducing subjectivity
- Citation accuracy is constrained by the free-tier model's reliability at inline citation
- High latency is a free-tier LLM constraint, not an application architecture issue
- System depends on retrieval quality from the ChromaDB vector store

## Future Improvements

- Add more evaluation questions
- Test different chunk sizes
- Compare top-k retrieval settings
- Add automated citation checking
- Add reranking for better retrieval precision
