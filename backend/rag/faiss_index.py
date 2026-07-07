from __future__ import annotations

import faiss
import numpy as np

from core.paths import EMBEDDING_FILE, FAISS_INDEX_FILE


class FaissIndexBuilder:
    def __init__(self) -> None:
        self.embeddings = None
        self.index = None

    def load_embeddings(self) -> None:
        print("Loading embedding vectors...")

        self.embeddings = np.load(EMBEDDING_FILE).astype("float32")

        print(f"Embedding Shape: {self.embeddings.shape}")

    def build_index(self) -> None:
        dimension = self.embeddings.shape[1]

        print(f"Building FAISS Index ({dimension} dimensions)...")

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        print(f"Indexed vectors: {self.index.ntotal}")

    def save_index(self) -> None:
        FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

        index_bytes = faiss.serialize_index(self.index)

        with FAISS_INDEX_FILE.open("wb") as file:
            file.write(index_bytes)

        print("FAISS index saved to:")
        print(FAISS_INDEX_FILE)

    def run(self) -> None:
        self.load_embeddings()
        self.build_index()
        self.save_index()

        print("FAISS Index Pipeline Completed.")


def main() -> None:
    builder = FaissIndexBuilder()
    builder.run()


if __name__ == "__main__":
    main()