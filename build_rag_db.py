"""
Run once to build the local ChromaDB from the bgGLUE fakenews dataset.
Usage:  python build_rag_db.py
"""
import re
from pathlib import Path

import chromadb
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

DB_PATH        = Path(__file__).parent / "rag_bg_news_db"
COLLECTION_NAME = "credible_bg_news"
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAMPLE_SIZE    = 700
BATCH_SIZE     = 32


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_document(row) -> str:
    return clean_text((row["title"] or "") + "\n\n" + (row["content"] or ""))


def main():
    print("Loading bgGLUE fakenews dataset...")
    dataset = load_dataset("bgglue/bgglue", "fakenews", trust_remote_code=True)
    full_data = concatenate_datasets([dataset["train"], dataset["validation"]])
    df = pd.DataFrame(full_data)

    credible_df = df[df["label"] == 0].copy()
    print(f"Credible articles: {len(credible_df)}")

    credible_df["rag_text"] = credible_df.apply(build_document, axis=1)
    credible_df = credible_df[credible_df["rag_text"].str.len() > 100].reset_index(drop=True)

    sample_df = credible_df.sample(
        n=min(SAMPLE_SIZE, len(credible_df)), random_state=42
    ).reset_index(drop=True)

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    documents = sample_df["rag_text"].tolist()
    metadatas = [
        {"title": str(row["title"]), "url": str(row["url"]), "source": "bgGLUE credible"}
        for _, row in sample_df.iterrows()
    ]
    ids = [f"bg_{i}" for i in range(len(sample_df))]

    print(f"Encoding {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=BATCH_SIZE)

    print(f"Writing to ChromaDB at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Drop existing collection so re-runs start clean
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"Done. Indexed {collection.count()} documents.")


if __name__ == "__main__":
    main()
