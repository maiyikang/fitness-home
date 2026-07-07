from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from core.paths import DATA_DIR, DATABASE_FILE


OUTPUT_FILE = DATA_DIR / "lora_training_data.jsonl"
MAX_SAMPLES = 100


def get_database_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def build_user_request(restaurant: Dict, menu_item: Dict) -> str:
    cuisine = restaurant.get("category", "restaurant")
    calories = menu_item.get("estimated_calories", 0)
    protein = menu_item.get("estimated_protein", 0)
    fiber = menu_item.get("estimated_fiber", 0)
    tags = menu_item.get("tags", "")

    if protein >= 35 and calories <= 600:
        return f"I want a high-protein {cuisine} meal under 600 calories."

    if fiber >= 8 and calories <= 600:
        return f"I want a high-fiber healthy {cuisine} meal under 600 calories."

    if calories <= 500:
        return f"I want a low-calorie {cuisine} meal under 500 calories."

    if "vegetarian" in tags.lower():
        return f"I want a vegetarian {cuisine} meal with good nutrition."

    return f"I want a fitness-friendly {cuisine} meal recommendation."


def build_training_output(restaurant: Dict, menu_item: Dict) -> str:
    restaurant_name = restaurant.get("name", "")
    menu_name = menu_item.get("name", "")
    calories = menu_item.get("estimated_calories", 0)
    protein = menu_item.get("estimated_protein", 0)
    fiber = menu_item.get("estimated_fiber", 0)
    health_level = menu_item.get("health_level", "normal")

    return (
        f"Recommended Restaurant:\n"
        f"{restaurant_name}\n\n"
        f"Recommended Menu:\n"
        f"{menu_name}\n\n"
        f"Nutrition Summary:\n"
        f"{calories} kcal, {protein}g protein, {fiber}g fiber\n\n"
        f"Recommended Reason:\n"
        f"{restaurant_name} is recommended because {menu_name} is a {health_level} option "
        f"with {protein}g protein, {fiber}g fiber, and {calories} kcal. "
        f"This makes it suitable for a fitness-oriented meal recommendation."
    )


def load_training_candidates() -> List[Dict]:
    connection = get_database_connection()

    query = """
        SELECT
            restaurants.id AS restaurant_id,
            restaurants.name AS restaurant_name,
            restaurants.category AS restaurant_category,
            restaurants.health_score AS restaurant_health_score,
            restaurants.fitness_score AS restaurant_fitness_score,
            menu_items.id AS menu_id,
            menu_items.name AS menu_name,
            menu_items.category AS menu_category,
            menu_items.estimated_calories AS estimated_calories,
            menu_items.estimated_protein AS estimated_protein,
            menu_items.estimated_fiber AS estimated_fiber,
            menu_items.health_level AS health_level,
            menu_items.tags AS tags
        FROM menu_items
        JOIN restaurants
            ON menu_items.restaurant_id = restaurants.id
        WHERE
            menu_items.estimated_calories IS NOT NULL
            AND menu_items.estimated_protein IS NOT NULL
            AND menu_items.estimated_fiber IS NOT NULL
            AND menu_items.estimated_calories <= 700
            AND menu_items.estimated_protein >= 18
        ORDER BY
            menu_items.estimated_protein DESC,
            menu_items.estimated_fiber DESC,
            menu_items.estimated_calories ASC
        LIMIT ?
    """

    cursor = connection.cursor()
    cursor.execute(query, (MAX_SAMPLES,))
    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


def build_dataset() -> List[Dict[str, str]]:
    candidates = load_training_candidates()
    dataset: List[Dict[str, str]] = []

    for item in candidates:
        restaurant = {
            "id": item["restaurant_id"],
            "name": item["restaurant_name"],
            "category": item["restaurant_category"],
            "health_score": item["restaurant_health_score"],
            "fitness_score": item["restaurant_fitness_score"],
        }

        menu_item = {
            "id": item["menu_id"],
            "name": item["menu_name"],
            "category": item["menu_category"],
            "estimated_calories": item["estimated_calories"],
            "estimated_protein": item["estimated_protein"],
            "estimated_fiber": item["estimated_fiber"],
            "health_level": item["health_level"],
            "tags": item["tags"],
        }

        sample = {
            "instruction": "Recommend a fitness meal based on the user's nutrition request.",
            "input": build_user_request(restaurant, menu_item),
            "output": build_training_output(restaurant, menu_item),
        }

        dataset.append(sample)

    return dataset


def save_dataset(dataset: List[Dict[str, str]]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for sample in dataset:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    print("Building LoRA training dataset...")

    dataset = build_dataset()
    save_dataset(dataset)

    print(f"Created training samples: {len(dataset)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("LoRA dataset building completed.")


if __name__ == "__main__":
    main()