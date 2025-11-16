import os
from typing import List, Tuple

import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from utils import get_project_paths
from run_retrieval import load_artifacts, MODEL_NAME, TOP_K_CHUNKS, TOP_N_DOCS


# Пример cross-encoder'а для русско-английского текста.
# При желании модель можно заменить на более подходящую.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_models() -> Tuple[SentenceTransformer, CrossEncoder]:
    bi_encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    return bi_encoder, cross_encoder


def rerank_for_query(
    query: str,
    bi_encoder: SentenceTransformer,
    cross_encoder: CrossEncoder,
    index,
    chunks_df: pd.DataFrame,
    top_k_chunks: int = TOP_K_CHUNKS,
    top_n_docs: int = TOP_N_DOCS,
) -> List[int]:
    """
    Двухэтапный поиск:
    1) bi-encoder + FAISS возвращает top_k_chunks кандидатов;
    2) cross-encoder переоценивает пары (query, text_chunk), затем мы агрегируем
       cross-encoder-счёты на уровне документов и выбираем top_n_docs.
    """
    query_emb = bi_encoder.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    scores, indices = index.search(query_emb.astype("float32"), top_k_chunks)
    scores = scores[0]
    indices = indices[0]

    # Формируем пары (query, text_chunk) для rerank
    pairs = []
    meta: List[Tuple[int, int]] = []  # (web_id, chunk_id)
    for idx in indices:
        row = chunks_df.iloc[int(idx)]
        text_chunk = str(row["text_chunk"])
        web_id = int(row["web_id"])
        chunk_id = int(row["chunk_id"])
        pairs.append((query, text_chunk))
        meta.append((web_id, chunk_id))

    if not pairs:
        return []

    ce_scores = cross_encoder.predict(pairs)

    # Агрегируем по web_id, берём максимум cross-encoder-счёта среди чанков
    doc_scores = {}
    for (web_id, _chunk_id), sc in zip(meta, ce_scores):
        prev = doc_scores.get(web_id, float("-inf"))
        if sc > prev:
            doc_scores[web_id] = float(sc)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_web_ids = [doc_id for doc_id, _ in sorted_docs[:top_n_docs]]
    return top_web_ids


if __name__ == "__main__":
    """
    Пример использования rerank'ера для одного запроса.
    Для интеграции в пайплайн можно импортировать `rerank_for_query`
    и применять его вместо простого aggregate_scores_by_web_id.
    """
    paths = get_project_paths()
    data_dir = paths["data_dir"]

    questions_path = os.path.join(data_dir, "questions_clean.csv")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"questions_clean.csv not found at {questions_path}")

    print("Loading questions...")
    questions_df = pd.read_csv(questions_path)
    query_example = str(questions_df.iloc[0]["query"])
    print("Example query:", query_example)

    print("Loading index and metadata ...")
    index, chunks_df = load_artifacts()

    print("Loading models ...")
    bi_encoder, cross_encoder = load_models()

    print("Running rerank for example query ...")
    top_web_ids = rerank_for_query(
        query_example,
        bi_encoder,
        cross_encoder,
        index,
        chunks_df,
    )
    print("Top web_ids after rerank:", top_web_ids)


