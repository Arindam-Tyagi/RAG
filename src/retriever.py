

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="./data/chroma_db"
)

collection = client.get_collection(
    name="pdf_chunks"
)

# Use the SAME model used to embed your chunks
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def retrieve(question, top_k=3):
    # 1. Convert question into a vector
    query_embedding = model.encode(question).tolist()

    # 2. Search the vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    # 3. Return matching chunks and distances
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]

    for i, (chunk, distance) in enumerate(
        zip(retrieved_chunks, distances), start=1
    ):
        print(f"\nResult {i} | Distance: {distance:.4f}")
        print(chunk)

    return retrieved_chunks


if __name__ == "__main__":
    question = input("Ask a question: ")
    retrieve(question)
