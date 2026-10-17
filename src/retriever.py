
import re


import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --------------------------------------------------

DB_DIR = "./data/chroma_db"
COLLECTION_NAME = "pdf_chunks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieve extra candidates before combining rankings.
CANDIDATE_MULTIPLIER = 5
MIN_CANDIDATES = 20

# RRF rank-fusion constant.
RRF_K = 60


client = chromadb.PersistentClient(
    path=DB_DIR
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

model = SentenceTransformer(MODEL_NAME)


# 3. Tokenize text for BM25
# --------------------------------------------------

def tokenize(text):
    """
    Lowercase text and extract alphanumeric tokens.

    This baseline is intentionally simple. For technical
    identifiers, we can later add identifier-aware rules.
    """
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# 4. Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(rankings, k=RRF_K):
    """
    Combine multiple ranked lists using Reciprocal Rank
    Fusion. Each ranking contains chunk IDs, best first.
    """
    scores = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    return ranked_ids, scores



# --------------------------------------------------
# 5. Hybrid retrieval
# --------------------------------------------------

def retrieve(question, top_k=3, verbose=True):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Read the indexed corpus. We rebuild the BM25 index
    # per call so that changes to ChromaDB are reflected.
    stored = collection.get(
        include=["documents", "metadatas"]
    )

    # Keep IDs, text and metadata aligned.
    records = [
        (chunk_id, text, metadata)
        for chunk_id, text, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
        )
        if text and text.strip()
    ]

    if not records:
        if verbose:
            print("No indexed chunks found. Run main.py first.")
        return []

    ids = [record[0] for record in records]
    texts = [record[1] for record in records]
    metadata_by_id = {
        record[0]: record[2] for record in records
    }
    text_by_id = dict(zip(ids, texts))

    candidate_k = min(
        len(ids),
        max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES),
    )

    # ----------------------------------------------
    # A. Dense retrieval
    # ----------------------------------------------

    # Convert the question to a vector in the same
    # embedding space as the indexed chunk vectors.
    query_embedding = model.encode(
        question,
        normalize_embeddings=False,
    ).tolist()

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, collection.count()),
        include=["documents", "distances"],
    )

    dense_ids = dense_results["ids"][0]

    # ----------------------------------------------
    # B. Sparse retrieval using BM25
    # ----------------------------------------------

    tokenized_corpus = [
        tokenize(text) for text in texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(question)

    if query_tokens:
        bm25_scores = bm25.get_scores(query_tokens)

        # Sort by descending BM25 relevance score.
        sparse_indices = sorted(
            range(len(ids)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:candidate_k]

        sparse_ids = [ids[i] for i in sparse_indices]
    else:
        sparse_ids = []

    # ----------------------------------------------
    # C. Fuse dense and sparse rankings
    # ----------------------------------------------

    fused_ids, fused_scores = reciprocal_rank_fusion(
        [dense_ids, sparse_ids]
    )

    final_ids = fused_ids[:top_k]

    retrieved_chunks = [
        text_by_id[chunk_id]
        for chunk_id in final_ids
        if chunk_id in text_by_id
    ]

    # ----------------------------------------------
    # D. Display diagnostics
    # ----------------------------------------------

    if verbose:
        print(f"\nQuestion: {question}")
        print(f"Indexed chunks: {len(ids)}")
        print(f"Dense candidates: {len(dense_ids)}")
        print(f"BM25 candidates: {len(sparse_ids)}")
        print("\n========== HYBRID RESULTS ==========")

        for rank, chunk_id in enumerate(final_ids, start=1):
            print(f"\nResult {rank}")
            print(f"Chunk ID: {chunk_id}")
            print(
                f"RRF score: {fused_scores[chunk_id]:.6f}"
            )
            print(
                f"Metadata: {metadata_by_id.get(chunk_id)}"
            )
            print(text_by_id[chunk_id])

        print("\n===================================")

    return retrieved_chunks


# --------------------------------------------------
# 6. Run directly from the terminal
# --------------------------------------------------

if __name__ == "__main__":
    question = input("Ask a question: ")
    retrieve(question, top_k=3)
