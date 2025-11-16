import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from utils import (
    preprocess_text,
    normalize_embeddings,
    get_project_paths,
)


# Bi-encoder for retrieval (should match model in build_index.py)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K_CHUNKS = 50
TOP_N_DOCS = 5


def load_artifacts() -> Tuple[faiss.Index, pd.DataFrame]:
    paths = get_project_paths()
    artifacts_dir = paths["artifacts_dir"]

    index_path = os.path.join(artifacts_dir, "faiss_index.bin")
    meta_path = os.path.join(artifacts_dir, "chunks_metadata.csv")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Chunks metadata not found: {meta_path}")

    # Use Python IO + deserialize_index with numpy buffer
    # to avoid issues with non-ASCII paths on Windows
    with open(index_path, "rb") as f:
        data = f.read()
    data_array = np.frombuffer(data, dtype="uint8")
    index = faiss.deserialize_index(data_array)
    chunks_df = pd.read_csv(meta_path)
    return index, chunks_df


def aggregate_scores_by_web_id(
    indices: np.ndarray,
    scores: np.ndarray,
    chunks_df: pd.DataFrame,
    top_n_docs: int = TOP_N_DOCS,
) -> List[int]:
    """
    Aggregate chunk-level scores to document-level scores and return top web_id list.
    """
    doc_scores: Dict[int, float] = defaultdict(float)

    for idx, score in zip(indices, scores):
        row = chunks_df.iloc[int(idx)]
        web_id = int(row["web_id"])
        # Sum scores across chunks of the same document
        doc_scores[web_id] += float(score)

    # Sort by aggregated score descending
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_web_ids = [doc_id for doc_id, _ in sorted_docs[:top_n_docs]]
    return top_web_ids


def main() -> None:
    paths = get_project_paths()
    data_dir = paths["data_dir"]

    questions_path = os.path.join(data_dir, "questions_clean.csv")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"questions_clean.csv not found at {questions_path}")

    print(f"Loading questions from {questions_path}")
    questions_df = pd.read_csv(questions_path)

    print("Loading FAISS index and metadata ...")
    index, chunks_df = load_artifacts()
    dim = index.d
    print(f"Index dimension: {dim}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    results: List[Tuple[int, List[int]]] = []

    for _, row in questions_df.iterrows():
        q_id = int(row["q_id"])
        query = preprocess_text(str(row["query"]))
        if not query:
            # If query is empty, return empty list (will be handled later)
            results.append((q_id, []))
            continue

        query_emb = model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        query_emb = normalize_embeddings(query_emb)

        scores, indices = index.search(query_emb, TOP_K_CHUNKS)
        scores = scores[0]
        indices = indices[0]

        top_web_ids = aggregate_scores_by_web_id(indices, scores, chunks_df, TOP_N_DOCS)
        results.append((q_id, top_web_ids))

    # Build submission dataframe
    out_rows = []
    for q_id, web_ids in results:
        # If for some reason less than TOP_N_DOCS, just use what we have.
        web_ids = web_ids[:TOP_N_DOCS]
        web_list_str = "[" + ", ".join(str(int(w)) for w in web_ids) + "]"
        out_rows.append({"q_id": q_id, "web_list": web_list_str})

    submit_df = pd.DataFrame(out_rows)

    output_path = os.path.join(paths["artifacts_dir"], "submit.csv")
    print(f"Saving submission to {output_path}")
    submit_df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()


