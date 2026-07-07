from __future__ import annotations

import numpy as np

from core.paths import EMBEDDING_FILE


class VectorStore:

    def __init__(self) -> None:
        self.embedding_file = EMBEDDING_FILE

    def exists(self) -> bool:
        return self.embedding_file.exists()

    def load_embeddings(self) -> np.ndarray:

        if not self.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {self.embedding_file}"
            )

        embeddings = np.load(self.embedding_file)

        print(f"Loaded embedding matrix: {embeddings.shape}")

        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> None:

        np.save(
            self.embedding_file,
            embeddings,
        )

        print(f"Saved embeddings to:")

        print(self.embedding_file)

    def get_vector_dimension(self) -> int:

        embeddings = self.load_embeddings()

        return embeddings.shape[1]

    def get_document_count(self) -> int:

        embeddings = self.load_embeddings()

        return embeddings.shape[0]


def main():

    store = VectorStore()

    if not store.exists():

        print("Embedding file does not exist.")

        return

    print("Embedding Statistics")

    print("-------------------------")

    print(f"Documents : {store.get_document_count()}")

    print(f"Dimensions: {store.get_vector_dimension()}")


if __name__ == "__main__":

    main()