# Design and Evaluation

## Architecture Overview

For this project, I implemented a Retrieval-Augmented Generation (RAG) application using a straightforward architecture. Policy documents are ingested, split into chunks, and stored in a vector database. When a user submits a question, the application retrieves the most relevant document chunks and provides them to the language model as context for generating an answer.

The application is built with Flask and exposes both a web interface and API endpoints. My goal was to create a solution that was simple, cost-effective, and easy to understand while still demonstrating the core principles of a modern RAG system.

## Design Decisions

### Embedding Model — `all-MiniLM-L6-v2` (ONNX)

I selected ChromaDB's built-in `ONNXMiniLM_L6_V2` embedding model because it runs locally, requires no external API calls, and has no usage costs. The model is widely used for semantic similarity tasks and produces compact embeddings that work well for a relatively small document corpus.

Running embeddings locally also eliminated concerns about API quotas, network latency, and ongoing operating costs. Since the same model is used for both indexing and retrieval, consistency across the pipeline is maintained.

### Chunking Strategy — 900 characters, 150-character overlap, 100-character minimum

Documents are divided into chunks of approximately 900 characters with a 150-character overlap between chunks. A minimum chunk length of 100 characters is enforced to prevent degenerate end-of-section fragments from being indexed.

I chose this approach to balance context preservation with retrieval accuracy. Larger chunks can contain multiple unrelated concepts, which can weaken embedding quality. Smaller chunks can break policy statements apart and make it harder to retrieve complete answers. The overlap helps ensure important information is not lost at chunk boundaries. The minimum length filter prevents repetitive boilerplate fragments — such as short "Questions should be directed to HR" footers.

### Retrieval Strategy — Top-k = 4

The application retrieves the four most relevant chunks for each query.

In testing, four chunks provided enough context to answer most policy-related questions without unnecessarily increasing prompt size. Since the document collection is relatively small, retrieving four chunks also increases the likelihood of capturing supporting information from multiple policies when needed.

### Vector Store — ChromaDB

I selected ChromaDB because it is free, lightweight, and easy to integrate into a Python application. It also persists data locally, allowing the vector index to survive application restarts and deployments.

For a project of this size, ChromaDB provides all the functionality needed without introducing the complexity or cost of a managed vector database service.

### LLM Selection — OpenRouter Free Tier

The application uses OpenRouter's free-tier API with the `liquid/lfm-2.5-1.2b-thinking:free` model.

My primary objective was to build the entire solution using free or low-cost technologies. OpenRouter provided access to a capable language model without requiring paid API usage.

The tradeoff is performance. While answer quality was generally acceptable, latency was significantly higher than what would be expected from a paid model hosted on dedicated infrastructure.

### Prompt Design

The prompt structure consists of:

* A system prompt that establishes behavior and response rules
* A user prompt that contains the question and retrieved context

Retrieved chunks are numbered so the model can generate citations that map back to the supporting source material.

The prompt also includes several guardrails:

* Answer only using retrieved content
* Avoid answering questions outside the document corpus
* Include citations in responses
* Keep answers concise and focused
* Restrict citations to the numbered range actually provided 

These controls helped improve groundedness and reduce hallucinations.

### Application Framework — Flask

I chose Flask because it is lightweight, easy to understand, and well suited for a project of this scope.

Rather than introducing additional frameworks such as LangChain, I implemented the retrieval and prompt construction logic directly. This approach reduced dependencies and made it easier to understand exactly how data moved through the system.

## Evaluation Approach

To evaluate system performance, I created a set of 18 test questions covering:

* Paid Time Off (PTO)
* Travel policies
* Expense reimbursement
* Remote work
* Information security
* Out-of-scope topics

The evaluation focused on three key areas:

* Groundedness
* Citation Accuracy
* Latency

Groundedness measures whether answers are supported by retrieved evidence. Citation accuracy measures whether the cited source actually supports the answer provided. Latency measures total response time from request to answer.

## Evaluation Results

| Metric | Result |
|---|---:|
| Number of Questions | 18 |
| Groundedness | 86.1% |
| Citation Accuracy | 61.1% |
| Latency p50 | 25.97 s |
| Latency p95 | 33.19 s |

## Scoring Methodology

Groundedness and citation accuracy were manually scored using the following scoring scale:

* 1.0 = correct
* 0.5 = partially correct
* 0.0 = incorrect

Latency was measured automatically using Python's `time.perf_counter()`.

## Key Findings 

Overall, the application performed well on straightforward policy questions where the answer existed clearly within a single document.

Groundedness was the strongest area of performance. The retrieval process generally returned relevant content, and the prompt instructions helped keep the model focused on the provided evidence.

Latency was the most significant weakness. Retrieval and embedding operations were typically completed in under one second. Most of the response time was spent waiting for the language model. Because the application relies on a free-tier API, requests are subject to shared infrastructure and queue delays.

## Known Limitations

Several limitations should be considered when interpreting the results:

* The evaluation set included only 18 questions.
* Groundedness and citation accuracy were scored manually, which introduces some subjectivity.
* Response latency is heavily influenced by OpenRouter's free-tier infrastructure.
* Overall answer quality remains dependent on retrieval quality from the vector database.

## Future Improvements

If I continue developing this project, I would focus on the following enhancements:

* Expand the evaluation dataset.
* Experiment with different chunk sizes and overlap settings.
* Compare alternative top-k retrieval configurations.
* Implement automated citation validation.
* Add reranking to improve retrieval precision.
* Evaluate higher-performing language models to improve citation quality and reduce latency.

While there are opportunities for improvement, the project successfully demonstrates a complete end-to-end RAG architecture using entirely free-tier technologies and provides a strong foundation for future enhancements.
