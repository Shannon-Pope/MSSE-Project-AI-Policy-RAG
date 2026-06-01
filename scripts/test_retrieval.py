import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


CHROMA_DIR = "vectorstore/chroma"
COLLECTION_NAME = "company_policies"


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),  # type: ignore[arg-type]
    )

    query = input("Enter a policy question: ")

    n_results = min(3, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    print("\nTop retrieval results:\n")

    documents = (results["documents"] or [[]])[0]
    metadatas = (results["metadatas"] or [[]])[0]
    distances = (results["distances"] or [[]])[0]

    if not documents:
        print("No results found.")
        return

    for i, doc in enumerate(documents, start=1):
        meta = metadatas[i - 1] or {}
        distance = distances[i - 1]

        print(f"Result {i}")
        print(f"Document: {meta.get('document_title')}")
        print(f"File: {meta.get('file_name')}")
        print(f"Page: {meta.get('page')}")
        print(f"Chunk ID: {meta.get('chunk_id')}")
        print(f"Distance: {distance:.4f}  (lower = more similar)")
        print("Snippet:")
        print(doc[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
