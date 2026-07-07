from __future__ import annotations

import json
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL, NORMALIZE_EMBEDDINGS
from core.paths import DOCUMENT_FILE, EMBEDDING_FILE


class EmbeddingBuilder:
    def __init__(self) -> None:
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def load_documents(self) -> List[dict]:
        documents: List[dict] = []

        with DOCUMENT_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                documents.append(json.loads(line))

        return documents

    def build_embeddings(self, documents: List[dict]) -> np.ndarray:
        texts = [document["text"] for document in documents]

        print(f"Encoding {len(texts)} documents...")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray) -> None:
        np.save(EMBEDDING_FILE, embeddings)

        print("Embedding file saved to:")
        print(EMBEDDING_FILE)

    def run(self) -> None:
        documents = self.load_documents()

        print(f"Loaded documents: {len(documents)}")

        embeddings = self.build_embeddings(documents)

        print(f"Embedding shape: {embeddings.shape}")

        self.save_embeddings(embeddings)

        print("Embedding pipeline completed.")


def main() -> None:
    builder = EmbeddingBuilder()
    builder.run()


if __name__ == "__main__":
    main()