# Design and Evaluation

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

The system performed the best on direct policy questions, where the answer was explicitly in the source documents.

The system performed weaker when questions required interpretion of multiple policies or when retrieved chunks were only loosely related.

## Known Limitations

- Evaluation set is small
- Groundedness and citation accuracy were manually scored
- Latency may vary, depending on LLM API response time
- System depends on retrieval quality from the Chroma vector store

## Future Improvements

- Add more evaluation questions
- Test different chunk sizes
- Compare top-k retrieval settings
- Add automated citation checking
- Add reranking for better retrieval precision
