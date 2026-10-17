
# src/chunker.py

import hashlib
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 100
BREAKPOINT_PERCENTILE = 75
BATCH_SIZE = 32

# The model is used to detect semantic boundaries between sentences.
semantic_model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if OVERLAP_TOKENS >= MAX_CHUNK_TOKENS:
    raise ValueError("OVERLAP_TOKENS must be smaller than MAX_CHUNK_TOKENS")


# --------------------------------------------------
# 1. Token utilities
# --------------------------------------------------

def encode_text(text):
    """Convert text to tokenizer token IDs."""
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def decode_tokens(token_ids):
    """Convert token IDs back to readable text."""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def count_tokens(text):
    """Count tokens using the configured tokenizer."""
    return len(encode_text(text))


# --------------------------------------------------
# 2. Split oversized sentences using tokens
# --------------------------------------------------

def split_long_sentence(sentence, max_tokens=MAX_CHUNK_TOKENS):
    """
    Split a sentence into token-limited pieces.
    Preserve its original page number.
    """
    text = sentence["text"].strip()

    if not text:
        return []

    token_ids = encode_text(text)

    if len(token_ids) <= max_tokens:
        return [{
            "text": text,
            "page": sentence["page"],
        }]

    pieces = []

    for start in range(0, len(token_ids), max_tokens):
        piece_ids = token_ids[start:start + max_tokens]
        piece_text = decode_tokens(piece_ids)

        if piece_text:
            pieces.append({
                "text": piece_text,
                "page": sentence["page"],
            })

    return pieces


# --------------------------------------------------
# 3. Extract sentences from PDF pages
# --------------------------------------------------

def split_sentences(pages):
    """
    Input:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."}
        ]

    Output:
        [
            {"text": "First sentence.", "page": 1},
            ...
        ]

    Sentence splitting is rule-based. Semantic grouping
    is performed separately using sentence embeddings.
    """
    sentences = []

    # Sentence-ending punctuation followed by whitespace
    # or a line break.
    sentence_pattern = re.compile(
        r'(?<=[.!?])\s+|[\r\n]+'
    )

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data.get("text", "")

        if not page_text or not page_text.strip():
            continue

        parts = sentence_pattern.split(page_text)

        for part in parts:
            text = re.sub(r"\s+", " ", part).strip()

            if not text:
                continue

            sentence_parts = split_long_sentence({
                "text": text,
                "page": page_number,
            })

            sentences.extend(sentence_parts)

    return sentences


# --------------------------------------------------
# 4. Calculate semantic distance
# --------------------------------------------------

def find_semantic_groups(sentences):
    """
    Group adjacent sentences based on embedding similarity.

    A larger cosine distance between adjacent sentences
    indicates a possible semantic boundary.

    Returns:
        [
            [sentence_1, sentence_2],
            [sentence_3, sentence_4],
        ]
    """
    if not sentences:
        return []

    if len(sentences) == 1:
        return [sentences]

    texts = [sentence["text"] for sentence in sentences]

    embeddings = semantic_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Because embeddings are normalized, their dot product
    # equals cosine similarity.
    similarities = np.sum(
        embeddings[:-1] * embeddings[1:],
        axis=1,
    )

    distances = 1.0 - similarities

    threshold = float(
        np.percentile(distances, BREAKPOINT_PERCENTILE)
    )

    groups = []
    current_group = [sentences[0]]

    for index, distance in enumerate(distances):
        next_sentence = sentences[index + 1]

        if distance > threshold:
            groups.append(current_group)
            current_group = [next_sentence]
        else:
            current_group.append(next_sentence)

    if current_group:
        groups.append(current_group)

    return groups


# --------------------------------------------------
# 5. Build token-limited chunks with overlap
# --------------------------------------------------

def build_chunks(semantic_groups, source_name):
    """
    Convert semantic groups into chunks.

    MAX_CHUNK_TOKENS includes overlap.
    OVERLAP_TOKENS are copied from the previous chunk's
    content to preserve context during retrieval.

    Returns chunk dictionaries ready for JSON storage
    and ChromaDB indexing.
    """
    chunks = []

    # Reserve space for overlap so the final chunk stays
    # within MAX_CHUNK_TOKENS.
    core_limit = MAX_CHUNK_TOKENS - OVERLAP_TOKENS

    previous_core_tokens = []

    chunk_index = 0

    for group_index, group in enumerate(semantic_groups):
        # Preserve page information alongside every token.
        group_tokens = []

        for sentence in group:
            token_ids = encode_text(sentence["text"])

            for token_id in token_ids:
                group_tokens.append({
                    "token_id": token_id,
                    "page": sentence["page"],
                })

        if not group_tokens:
            continue

        # A semantic group may be much larger than one chunk.
        # Split it into token-limited core sections.
        for start in range(0, len(group_tokens), core_limit):
            core = group_tokens[start:start + core_limit]

            if not core:
                continue

            core_token_ids = [
                item["token_id"] for item in core
            ]

            # Overlap from the preceding core. Keep overlap
            # even when a new semantic group starts.
            overlap = previous_core_tokens[-OVERLAP_TOKENS:]

            final_token_ids = overlap + core_token_ids
            final_text = decode_tokens(final_token_ids)

            if not final_text:
                previous_core_tokens = core_token_ids
                continue

            # Use core pages for metadata. The overlap may
            # also include text from an earlier page.
            core_pages = [item["page"] for item in core]

            page_start = min(core_pages)
            page_end = max(core_pages)

            chunk_hash = hashlib.sha256(
                (
                    f"{source_name}|{chunk_index}|"
                    f"{final_text}"
                ).encode("utf-8")
            ).hexdigest()[:16]

            actual_token_count = count_tokens(final_text)

            if actual_token_count > MAX_CHUNK_TOKENS:
                # Defensive check in case decoding and
                # re-encoding changes token boundaries.
                final_token_ids = final_token_ids[
                    -MAX_CHUNK_TOKENS:
                ]
                final_text = decode_tokens(final_token_ids)
                actual_token_count = count_tokens(final_text)

            chunks.append({
                "chunk_id": chunk_hash,
                "text": final_text,
                "token_count": actual_token_count,
                "metadata": {
                    "source": source_name,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": chunk_index,
                    "semantic_group": group_index,
                    "embedding_model": MODEL_NAME,
                    "chunking_method": "semantic",
                },
            })

            chunk_index += 1
            previous_core_tokens = core_token_ids

    return chunks


# --------------------------------------------------
# 6. Convenience function for a single document
# --------------------------------------------------

def create_chunks(pages, source_name="unknown.pdf"):
    """
    End-to-end chunking function.

    Use this from main.py:

        chunks = create_chunks(
            doc["pages"],
            source_name=doc["filename"]
        )
    """
    sentences = split_sentences(pages)

    if not sentences:
        return []

    semantic_groups = find_semantic_groups(sentences)

    return build_chunks(
        semantic_groups,
        source_name=source_name,
    )
