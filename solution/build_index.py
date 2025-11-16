import os
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

from utils import (
    build_chunks_dataframe,
    normalize_embeddings,
    get_project_paths,
)


# Use already cached multilingual model to avoid long downloads.
# При желании модель можно сменить, но важно использовать ту же самую в run_retrieval.py.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 64

# Параметры чанкинга подобраны чуть агрессивнее, чем по умолчанию:
# - меньший размер чанка позволяет точнее «прицелиться» в релевантный фрагмент,
# - достаточный overlap сохраняет контекст.
MAX_CHARS = 800
OVERLAP_CHARS = 160


def load_websites(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "websites.csv")
    df = pd.read_csv(path)
    # Ensure required columns
    expected_cols = {"web_id", "url", "kind", "title", "text"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in websites.csv: {missing}")
    return df


def encode_chunks(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    all_embeddings: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding chunks"):
        batch_texts = texts[start : start + batch_size]
        embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(embeddings.astype("float32"))
    if not all_embeddings:
        return np.zeros((0, 768), dtype="float32")
    return np.vstack(all_embeddings)


def main() -> None:
    paths = get_project_paths()
    data_dir = paths["data_dir"]
    artifacts_dir = paths["artifacts_dir"]

    print(f"Data directory: {data_dir}")
    print("Loading websites.csv ...")
    websites_df = load_websites(data_dir)

    print("Building chunks dataframe ...")
    chunks_df = build_chunks_dataframe(
        websites_df,
        max_chars=MAX_CHARS,
        overlap_chars=OVERLAP_CHARS,
    )
    print(f"Total chunks: {len(chunks_df)}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Encoding chunks to embeddings ...")
    embeddings = encode_chunks(model, chunks_df["text_chunk"].tolist())
    print(f"Embeddings shape: {embeddings.shape}")

    print("Normalizing embeddings ...")
    embeddings = normalize_embeddings(embeddings)

    dim = embeddings.shape[1]
    print(f"Building FAISS index with dim={dim} ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors.")

    index_path = os.path.join(artifacts_dir, "faiss_index.bin")
    meta_path = os.path.join(artifacts_dir, "chunks_metadata.csv")

    print(f"Saving FAISS index to {index_path}")
    # On Windows with non-ASCII paths FAISS C++ IO can fail,
    # so use Python IO + serialize_index instead.
    serialized_index = faiss.serialize_index(index)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        f.write(serialized_index)

    print(f"Saving chunks metadata to {meta_path}")
    chunks_df.to_csv(meta_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()


