from __future__ import annotations

from typing import Any, Dict, List

from lora.inference import LoraInference
from lora.prompt_builder import build_recommendation_prompt
from rag.retriever import retrieve


class RagLoraPipeline:
    def __init__(self) -> None:
        self.inference = LoraInference()

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        retrieved_results = retrieve(
            query=query,
            top_k=top_k,
        )

        prompt = build_recommendation_prompt(
            user_query=query,
            retrieved_results=retrieved_results,
        )

        generated_answer = self.inference.generate(prompt)

        return {
            "query": query,
            "retrieved_count": len(retrieved_results),
            "retrieved_results": retrieved_results,
            "generated_answer": generated_answer,
        }


def main() -> None:
    pipeline = RagLoraPipeline()

    query = "I want a high-protein Japanese meal under 600 calories."

    result = pipeline.run(
        query=query,
        top_k=5,
    )

    print()
    print("RAG + Llama Result")
    print("==================")
    print(result["generated_answer"])


if __name__ == "__main__":
    main()