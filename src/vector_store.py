
import chromadb

# Create a persistent local database
client = chromadb.PersistentClient(
    path="./data/chroma_db"
)

# Create or open a collection
collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"}
)

def store_chunks(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must match")

    if not chunks:
        return

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
        if hasattr(embeddings, "tolist")
        else embeddings,
        metadatas=[
            {"chunk_index": i}
            for i in range(len(chunks))
        ]
    )

    print("Stored chunks:", collection.count())
