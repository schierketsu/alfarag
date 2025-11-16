import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd


def preprocess_text(text: str) -> str:
    """
    Basic text normalization:
    - convert to lower case
    - collapse multiple whitespaces
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    return text


def split_to_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter based on punctuation.
    Good enough for building chunks on Russian text.
    """
    if not text:
        return []
    # Keep punctuation as separators, then strip
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def build_chunks_from_text(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Build overlapping chunks from a long text.
    Uses a sliding window over characters, trying to cut on sentence boundaries.
    """
    text = text or ""
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        # Try to move end to the last sentence boundary within window
        window = text[start:end]
        last_punct = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
        if last_punct != -1 and last_punct > max_chars * 0.4:
            end = start + last_punct + 1
            window = text[start:end]

        chunks.append(window.strip())

        if end >= text_len:
            break

        # Move with overlap
        start = max(0, end - overlap_chars)

    return chunks


def build_chunks_dataframe(
    websites_df: pd.DataFrame,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> pd.DataFrame:
    """
    From websites dataframe build a dataframe of text chunks.
    Columns: chunk_id (int), web_id (int), url, title, text_chunk (str)
    """
    chunk_records: List[Tuple[int, int, str, str, str]] = []
    chunk_id = 0

    for _, row in websites_df.iterrows():
        web_id = int(row["web_id"])
        url = row.get("url", "")
        title = row.get("title", "")
        text = row.get("text", "")
        text = preprocess_text(str(text))
        chunks = build_chunks_from_text(text, max_chars=max_chars, overlap_chars=overlap_chars)

        for ch in chunks:
            chunk_records.append(
                (chunk_id, web_id, url, title, ch),
            )
            chunk_id += 1

    chunks_df = pd.DataFrame(
        chunk_records,
        columns=["chunk_id", "web_id", "url", "title", "text_chunk"],
    )
    return chunks_df


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings for cosine-similarity search with inner product index.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


def get_project_paths() -> dict:
    """
    Helper to get common paths used in scripts.
    Assumes this file is located in the `solution` folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "techzadanie"))
    artifacts_dir = current_dir
    return {
        "current_dir": current_dir,
        "data_dir": data_dir,
        "artifacts_dir": artifacts_dir,
    }


