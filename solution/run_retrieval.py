import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

from utils import (
    preprocess_text,
    normalize_embeddings,
    get_project_paths,
)



# Bi-encoder for retrieval (should match model in build_index.py)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Больше кандидатов на уровне чанков повышает шанс поймать релевантный документ,
# а финальный top-5 отберём уже после агрегации/гибридизации.
TOP_K_CHUNKS = 150
TOP_N_DOCS = 5

# Настройки гибридного поиска
USE_HYBRID_BM25 = True
BM25_TOP_K_CHUNKS = 150


def build_bm25_index(chunks_df: pd.DataFrame) -> Tuple[BM25Okapi, List[int]]:
    """
    Build a BM25 index over text_chunk for hybrid retrieval.

    Returns
    -------
    bm25 : BM25Okapi
        Fitted BM25 index.
    chunk_ids : List[int]
        List of chunk_id in the same order as in bm25.corpus.
    """
    # Используем простой токенайзер по пробелам: текст уже нормализован.
    tokenized_corpus: List[List[str]] = []
    chunk_ids: List[int] = []

    for _, row in chunks_df.iterrows():
        text = str(row["text_chunk"])
        tokens = text.split()
        tokenized_corpus.append(tokens)
        chunk_ids.append(int(row["chunk_id"]))

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunk_ids


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
    mode: str = "max",
) -> List[int]:
    """
    Aggregate chunk-level scores to document-level scores and return top web_id list.

    Parameters
    ----------
    mode: {"sum", "mean", "max"}
        - "sum": суммировать скор всех чанков документа (может переоценивать длинные документы)
        - "mean": усреднить по количеству чанков,
        - "max": взять лучший (максимальный) скор среди чанков, что часто даёт хороший баланс.
    """
    raw_scores: Dict[int, List[float]] = defaultdict(list)

    for idx, score in zip(indices, scores):
        row = chunks_df.iloc[int(idx)]
        web_id = int(row["web_id"])
        raw_scores[web_id].append(float(score))

    doc_scores: Dict[int, float] = {}
    for web_id, sc_list in raw_scores.items():
        if not sc_list:
            continue
        if mode == "sum":
            agg = float(np.sum(sc_list))
        elif mode == "mean":
            agg = float(np.mean(sc_list))
        else:  # "max" по умолчанию
            agg = float(np.max(sc_list))
        doc_scores[web_id] = agg

    # Sort by aggregated score descending
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_web_ids = [doc_id for doc_id, _ in sorted_docs[:top_n_docs]]
    return top_web_ids


def hybrid_faiss_bm25_retrieval(
    query: str,
    query_emb: np.ndarray,
    index: faiss.Index,
    chunks_df: pd.DataFrame,
    bm25: Optional[BM25Okapi],
    bm25_chunk_ids: Optional[List[int]],
    top_k_chunks: int = TOP_K_CHUNKS,
    top_n_docs: int = TOP_N_DOCS,
) -> List[int]:
    """
    Hybrid retrieval: комбинирует dense-скор из FAISS и BM25-скор по text_chunk.

    Стратегия:
    - берём top_k_chunks из FAISS по косинусной близости,
    - берём top_k_chunks из BM25 по токенам,
    - приводим всё к общему пространству документов (web_id),
    - нормализуем dense и BM25-скоры по [0, 1] и суммируем.
    """
    # --- 1. FAISS: dense retrieval ---
    faiss_scores, faiss_indices = index.search(query_emb, top_k_chunks)
    faiss_scores = faiss_scores[0]
    faiss_indices = faiss_indices[0]

    faiss_doc_scores: Dict[int, float] = defaultdict(float)
    for idx, score in zip(faiss_indices, faiss_scores):
        row = chunks_df.iloc[int(idx)]
        web_id = int(row["web_id"])
        faiss_doc_scores[web_id] = max(faiss_doc_scores[web_id], float(score))

    # --- 2. BM25: lexical retrieval ---
    bm25_doc_scores: Dict[int, float] = defaultdict(float)
    if bm25 is not None and bm25_chunk_ids is not None:
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Выберем top_k_chunks по BM25
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_chunks]
        for local_idx in top_bm25_indices:
            score = float(bm25_scores[local_idx])
            chunk_id = bm25_chunk_ids[local_idx]
            row = chunks_df[chunks_df["chunk_id"] == chunk_id].iloc[0]
            web_id = int(row["web_id"])
            bm25_doc_scores[web_id] = max(bm25_doc_scores[web_id], score)

    # --- 3. Нормализация и комбинирование ---
    all_web_ids = set(faiss_doc_scores.keys()) | set(bm25_doc_scores.keys())
    if not all_web_ids:
        return []

    def normalize_scores(score_dict: Dict[int, float]) -> Dict[int, float]:
        if not score_dict:
            return {}
        values = np.array(list(score_dict.values()), dtype="float32")
        min_v = float(values.min())
        max_v = float(values.max())
        if max_v - min_v < 1e-8:
            # все значения почти одинаковые
            return {k: 1.0 for k in score_dict.keys()}
        return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

    faiss_norm = normalize_scores(faiss_doc_scores)
    bm25_norm = normalize_scores(bm25_doc_scores)

    combined_scores: Dict[int, float] = {}
    for web_id in all_web_ids:
        dense_part = faiss_norm.get(web_id, 0.0)
        bm25_part = bm25_norm.get(web_id, 0.0)
        # Простая равновесная комбинация, веса можно подбирать на валидации.
        combined_scores[web_id] = dense_part + bm25_part

    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
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

    bm25 = None
    bm25_chunk_ids: Optional[List[int]] = None
    if USE_HYBRID_BM25:
        print("Building BM25 index over chunks ...")
        bm25, bm25_chunk_ids = build_bm25_index(chunks_df)

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

        if USE_HYBRID_BM25:
            top_web_ids = hybrid_faiss_bm25_retrieval(
                query=query,
                query_emb=query_emb,
                index=index,
                chunks_df=chunks_df,
                bm25=bm25,
                bm25_chunk_ids=bm25_chunk_ids,
                top_k_chunks=TOP_K_CHUNKS,
                top_n_docs=TOP_N_DOCS,
            )
        else:
            scores, indices = index.search(query_emb, TOP_K_CHUNKS)
            scores = scores[0]
            indices = indices[0]

            top_web_ids = aggregate_scores_by_web_id(
                indices,
                scores,
                chunks_df,
                top_n_docs=TOP_N_DOCS,
                mode="max",
            )
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


