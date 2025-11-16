import os
from typing import List, Dict, Sequence

import numpy as np
import pandas as pd

from utils import get_project_paths
from run_retrieval import (
    load_artifacts,
    MODEL_NAME,
    TOP_K_CHUNKS,
    TOP_N_DOCS,
    aggregate_scores_by_web_id,
)
from sentence_transformers import SentenceTransformer


def hit_at_k(
    y_true: Sequence[int],
    y_pred: Sequence[List[int]],
    k: int = 5,
) -> float:
    """
    Вычисление Hit@k.
    y_true: список истинных web_id (по одному на вопрос или любая релевантная метка),
    y_pred: список списков предсказанных web_id (top-k).
    """
    assert len(y_true) == len(y_pred)
    hits = 0
    for truth, preds in zip(y_true, y_pred):
        if int(truth) in set(preds[:k]):
            hits += 1
    return hits / len(y_true) if y_true else 0.0


def evaluate_from_labeled_csv(
    labeled_path: str,
    top_k_chunks: int = TOP_K_CHUNKS,
    top_n_docs: int = TOP_N_DOCS,
) -> float:
    """
    Оценка retriever'а по файлу с разметкой.

    Ожидается CSV с колонками:
        - q_id
        - query
        - web_id (или любая колонка с 'true_web_id')
    """
    paths = get_project_paths()

    print("Loading labeled data from:", labeled_path)
    df = pd.read_csv(labeled_path)

    if "web_id" not in df.columns:
        raise ValueError("Labeled file must contain 'web_id' column with ground truth.")

    print("Loading FAISS index and metadata ...")
    index, chunks_df = load_artifacts()

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    y_true: List[int] = []
    y_pred: List[List[int]] = []

    for _, row in df.iterrows():
        query = str(row["query"])
        true_web_id = int(row["web_id"])

        query_emb = model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = index.search(query_emb, top_k_chunks)
        scores = scores[0]
        indices = indices[0]

        top_web_ids = aggregate_scores_by_web_id(
            indices,
            scores,
            chunks_df,
            top_n_docs=top_n_docs,
            mode="max",
        )

        y_true.append(true_web_id)
        y_pred.append(top_web_ids)

    metric = hit_at_k(y_true, y_pred, k=top_n_docs)
    print(f"Hit@{top_n_docs}: {metric:.4f}")
    return metric


if __name__ == "__main__":
    paths = get_project_paths()
    # путь к файлу с разметкой пользователь может задать сам
    labeled_csv = os.path.join(paths["data_dir"], "labeled_questions.csv")
    if not os.path.exists(labeled_csv):
        raise FileNotFoundError(
            f"Expected labeled questions file at {labeled_csv}. "
            f"Provide a CSV with columns: q_id, query, web_id."
        )
    evaluate_from_labeled_csv(labeled_csv)


