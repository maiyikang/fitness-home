from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL, NORMALIZE_EMBEDDINGS, TOP_K
from core.paths import DATABASE_FILE, DOCUMENT_FILE, FAISS_INDEX_FILE


class RagRetriever:
    def __init__(self) -> None:
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("Loading FAISS index...")
        self.index = self.load_faiss_index()

        print("Loading RAG documents...")
        self.documents = self.load_documents()

    def load_faiss_index(self):
        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_FILE}")

        with FAISS_INDEX_FILE.open("rb") as file:
            index_bytes = file.read()

        index_array = np.frombuffer(index_bytes, dtype="uint8")
        return faiss.deserialize_index(index_array)

    def load_documents(self) -> List[Dict[str, Any]]:
        if not DOCUMENT_FILE.exists():
            raise FileNotFoundError(f"RAG document file not found: {DOCUMENT_FILE}")

        documents: List[Dict[str, Any]] = []

        with DOCUMENT_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                documents.append(json.loads(line))

        return documents

    def encode_query(self, query: str) -> np.ndarray:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )

        return query_embedding.astype("float32")

    def get_restaurant_from_database(
        self,
        restaurant_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not DATABASE_FILE.exists():
            return None

        connection = sqlite3.connect(DATABASE_FILE)
        connection.row_factory = sqlite3.Row

        try:
            cursor = connection.cursor()

            possible_table_names = [
                "restaurants",
                "restaurant",
                "restaurants_clean",
            ]

            for table_name in possible_table_names:
                try:
                    cursor.execute(
                        f"SELECT * FROM {table_name} WHERE id = ? LIMIT 1",
                        (restaurant_id,),
                    )

                    row = cursor.fetchone()

                    if row is not None:
                        return dict(row)

                except sqlite3.OperationalError:
                    continue

            return None

        finally:
            connection.close()

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.encode_query(query)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results: List[Dict[str, Any]] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            document = self.documents[index]
            restaurant_id = str(document.get("restaurant_id", ""))
            database_record = self.get_restaurant_from_database(restaurant_id)

            results.append(
                {
                    "rank": len(results) + 1,
                    "similarity_score": float(score),
                    "restaurant_id": restaurant_id,
                    "restaurant_name": document.get("restaurant_name", ""),
                    "category": document.get("category", ""),
                    "cuisine_tags": document.get("cuisine_tags", ""),
                    "average_calories": document.get("average_calories", ""),
                    "average_protein": document.get("average_protein", ""),
                    "average_fiber": document.get("average_fiber", ""),
                    "health_score": document.get("health_score", ""),
                    "fitness_score": document.get("fitness_score", ""),
                    "document_text": document.get("text", ""),
                    "database_record": database_record,
                }
            )

        return results


_retriever_instance: Optional[RagRetriever] = None


def get_retriever() -> RagRetriever:
    global _retriever_instance

    if _retriever_instance is None:
        _retriever_instance = RagRetriever()

    return _retriever_instance


def retrieve(
    query: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    retriever = get_retriever()
    return retriever.search(query=query, top_k=top_k)


def print_results(results: List[Dict[str, Any]]) -> None:
    print()
    print("Retrieval Results")
    print("=================")

    for result in results:
        print()
        print(f"Rank: {result['rank']}")
        print(f"Restaurant: {result['restaurant_name']}")
        print(f"Restaurant ID: {result['restaurant_id']}")
        print(f"Category: {result['category']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Average Calories: {result['average_calories']}")
        print(f"Average Protein: {result['average_protein']}")
        print(f"Average Fiber: {result['average_fiber']}")


def main() -> None:
    query = "I want a high-protein Japanese meal under 600 calories."

    print(f"Query: {query}")

    results = retrieve(query=query, top_k=TOP_K)

    print_results(results)


if __name__ == "__main__":
    main()