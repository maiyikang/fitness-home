from __future__ import annotations

from typing import Any, Dict, List


def build_recommendation_prompt(
    user_query: str,
    retrieved_results: List[Dict[str, Any]],
) -> str:
    if len(retrieved_results) == 0:
        return (
            "No suitable restaurant was retrieved. "
            "Explain that no reliable recommendation can be made."
        )

    best_result = retrieved_results[0]

    restaurant_name = best_result.get("restaurant_name", "")
    category = best_result.get("category", "")
    average_calories = best_result.get("average_calories", "")
    average_protein = best_result.get("average_protein", "")
    average_fiber = best_result.get("average_fiber", "")
    fitness_score = best_result.get("fitness_score", "")
    document_text = str(best_result.get("document_text", ""))[:800]

    prompt = f"""
You are a fitness meal recommendation assistant.

The recommendation has already been selected by the retrieval system.
You must not choose another restaurant.
You must only explain why the selected restaurant matches the user's request.

User Request:
{user_query}

Selected Restaurant:
{restaurant_name}

Category:
{category}

Nutrition Summary:
Calories: {average_calories} kcal
Protein: {average_protein} g
Fiber: {average_fiber} g
Fitness Score: {fitness_score}

Evidence:
{document_text}

Answer only with one paragraph.

Do not repeat the restaurant name.
Do not recommend another restaurant.
Explain only why the selected restaurant matches the user's request.

Reason:
""".strip()

    return prompt


def main() -> None:
    sample_results = [
        {
            "rank": 1,
            "restaurant_name": "Samurai",
            "category": "Japanese, Asian, Sushi",
            "average_calories": 579,
            "average_protein": 28,
            "average_fiber": 4,
            "fitness_score": 54,
            "document_text": (
                "Restaurant name: Samurai\n"
                "Representative menu items:\n"
                "- Hibachi Chicken: 350 kcal, 38g protein, 9g fiber, health level: healthy"
            ),
        }
    ]

    prompt = build_recommendation_prompt(
        user_query="I want a high-protein Japanese meal under 600 calories.",
        retrieved_results=sample_results,
    )

    print(prompt)


if __name__ == "__main__":
    main()