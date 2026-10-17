
import json
from pathlib import Path

import chromadb

from src.pdf_loader import load_pdfs
from src.embedder import create_embeddings
from src.retriever import retrieve

from src.chunker import (
    split_sentences,
    find_semantic_groups,
    build_chunks,
)


# -----------------------------
# Configuration
# -----------------------------

RAW_DIR = "data/raw"
CHUNKS_DIR = Path("data/chunks")
DB_DIR = Path("data/chroma_db")

COLLECTION_NAME = "pdf_chunks"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Setup ChromaDB
# -----------------------------

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# -----------------------------
# Load PDFs
# -----------------------------

documents = load_pdfs(RAW_DIR)

total_characters = 0
total_chunks = 0
total_embeddings = 0


# -----------------------------
# Process each PDF
# -----------------------------

for doc in documents:
    filename = doc["filename"]

    print(f"\nProcessing: {filename}")

    # Step 1: Count characters and retain page text
    for page in doc["pages"]:
        total_characters += len(page["text"])

    # Step 2: Split the document into sentences
    sentences = split_sentences(doc["pages"])

    if not sentences:
        print("No text found; skipping this PDF.")
        continue

    # Step 3: Group sentences semantically
    semantic_groups = find_semantic_groups(sentences)

    # Step 4: Build token-limited chunks with overlap
    document_chunks = build_chunks(
        semantic_groups,
        source_name=filename
    )

    if not document_chunks:
        print("No chunks generated; skipping this PDF.")
        continue

    # Step 5: Add embeddings
    texts = [
        chunk["text"]
        for chunk in document_chunks
    ]

    embeddings = create_embeddings(texts)

    if len(embeddings) != len(document_chunks):
        raise ValueError(
            f"Embedding count mismatch for {filename}"
        )

    # Step 6: Attach embeddings to JSON records
    for chunk, embedding in zip(
        document_chunks, embeddings
    ):
        chunk["embedding"] = embedding.tolist()

    total_chunks += len(document_chunks)
    total_embeddings += len(embeddings)

    # Step 7: Save chunks and embeddings together
    output_path = CHUNKS_DIR / (
        Path(filename).stem + ".json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            document_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Semantic groups: {len(semantic_groups)}")
    print(f"Chunks created: {len(document_chunks)}")
    print(f"Embeddings created: {len(embeddings)}")
    print(f"Saved to: {output_path}")

    # Step 8: Store in ChromaDB
    ids = [
        f"{Path(filename).stem}_{chunk['chunk_id']}"
        for chunk in document_chunks
    ]

    metadatas = []

    for chunk in document_chunks:
        metadata = chunk["metadata"]

        metadatas.append({
            "document": filename,
            "page_start": metadata["page_start"],
            "page_end": metadata["page_end"],
            "chunk_id": chunk["chunk_id"],
            "chunk_index": metadata["chunk_index"],
            "semantic_group": metadata["semantic_group"],
            "chunking_method": metadata["chunking_method"],
            "embedding_model": metadata["embedding_model"],
            "token_count": chunk["token_count"],
        })

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=[
            embedding.tolist()
            for embedding in embeddings
        ],
        metadatas=metadatas
    )

    print(f"Stored in ChromaDB: {len(ids)} chunks")


# -----------------------------
# Pipeline summary
# -----------------------------

print("\n========== PIPELINE SUMMARY ==========")
print("Total characters:", total_characters)
print("Total chunks:", total_chunks)
print("Total embeddings:", total_embeddings)
print("Chunks in ChromaDB:", collection.count())
print("ChromaDB location:", DB_DIR.resolve())
print("======================================")

# -----------------------------
# Test retrieval
# -----------------------------

question = "What is the summary or context?"
retrieve(question)
