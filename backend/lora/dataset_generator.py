from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import ollama

from core.paths import DATA_DIR, DATABASE_FILE


TEACHER_MODEL_NAME = "llama3.1:8b"

OUTPUT_FILE = DATA_DIR / "lora_training_data.jsonl"

METADATA_FILE = DATA_DIR / "lora_training_metadata.json"

DATASET_VERSION = "v2"

MAX_MENU_CANDIDATES = 1200

MAX_SAMPLES = 10000

QUERIES_PER_MENU = 8

RANDOM_SEED = 42

REQUEST_DELAY_SECONDS = 0.2


SYSTEM_PROMPT = """
You are a fitness nutrition recommendation expert.

You are generating supervised fine-tuning data for a restaurant recommendation system.

The restaurant and menu item have already been selected by a retrieval system.

You must only generate the recommendation explanation.

Rules:

1. Never recommend another restaurant.
2. Never replace the selected menu item.
3. Never invent calories, protein, fiber, health level, or tags.
4. Use only the provided evidence.
5. Write naturally and avoid repetitive wording.
6. Mention why the selected restaurant and menu match the user's goal.
7. Mention relevant nutrition facts when useful.
8. Keep the explanation concise.
9. Output only the explanation text.
10. Do not use markdown.
""".strip()


GOAL_TEMPLATES: Dict[str, List[str]] = {
    "fat_loss": [
        "I want a lower-calorie {cuisine} meal that still supports my fitness goal.",
        "Recommend a {cuisine} meal for fat loss with reasonable calories.",
        "I need a fitness-friendly {cuisine} meal that is not too heavy.",
        "I want a lighter {cuisine} option that can fit into a calorie-controlled diet.",
    ],
    "muscle_gain": [
        "I want a high-protein {cuisine} meal for muscle gain.",
        "Recommend a {cuisine} meal that supports strength training.",
        "I need a protein-focused {cuisine} meal after a workout.",
        "I want a {cuisine} meal that helps with muscle recovery.",
    ],
    "maintenance": [
        "I want a balanced {cuisine} meal for daily fitness maintenance.",
        "Recommend a {cuisine} meal with balanced nutrition.",
        "I need a healthy everyday {cuisine} meal.",
        "I want a {cuisine} meal that supports a stable fitness routine.",
    ],
    "high_fiber": [
        "I want a high-fiber {cuisine} meal for better nutrition.",
        "Recommend a {cuisine} meal with good fiber content.",
        "I need a healthy {cuisine} meal that includes fiber.",
        "I want a fiber-friendly {cuisine} meal.",
    ],
    "high_protein": [
        "I want a high-protein {cuisine} meal under 600 calories.",
        "Recommend a protein-rich {cuisine} meal.",
        "I need a {cuisine} meal with strong protein content.",
        "I want a fitness-friendly {cuisine} meal with high protein.",
    ],
    "healthy_eating": [
        "I want a healthy {cuisine} meal recommendation.",
        "Recommend a fitness-oriented {cuisine} meal.",
        "I need a nutritious {cuisine} meal option.",
        "I want a {cuisine} meal that fits a healthy lifestyle.",
    ],
    "post_workout": [
        "I want a post-workout {cuisine} meal.",
        "Recommend a {cuisine} meal after training.",
        "I need a recovery-friendly {cuisine} meal.",
        "I want a {cuisine} meal that works well after exercise.",
    ],
    "low_calorie": [
        "I want a low-calorie {cuisine} meal.",
        "Recommend a lighter {cuisine} meal option.",
        "I need a {cuisine} meal that is not too high in calories.",
        "I want a calorie-conscious {cuisine} meal.",
    ],
}


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def normalize_text(value: Optional[Any]) -> str:
    if value is None:
        return ""

    return str(value).strip()


def normalize_number(value: Optional[Any], default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default
    except TypeError:
        return default


def clean_cuisine(category: str) -> str:
    category = normalize_text(category)

    if category == "":
        return "restaurant"

    parts = [
        item.strip()
        for item in category.split(",")
        if item.strip()
    ]

    if len(parts) == 0:
        return "restaurant"

    return parts[0]


def load_candidates() -> List[Dict[str, Any]]:
    connection = get_connection()
    cursor = connection.cursor()

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
            menu_items.name IS NOT NULL
            AND restaurants.name IS NOT NULL
            AND menu_items.estimated_calories IS NOT NULL
            AND menu_items.estimated_protein IS NOT NULL
            AND menu_items.estimated_fiber IS NOT NULL
            AND menu_items.estimated_calories > 0
            AND menu_items.estimated_calories <= 900
            AND menu_items.estimated_protein >= 8

        ORDER BY RANDOM()

        LIMIT ?
    """

    cursor.execute(query, (MAX_MENU_CANDIDATES,))
    rows = cursor.fetchall()
    connection.close()

    candidates = [dict(row) for row in rows]

    return candidates


def infer_goals(item: Dict[str, Any]) -> List[str]:
    calories = normalize_number(item.get("estimated_calories"))
    protein = normalize_number(item.get("estimated_protein"))
    fiber = normalize_number(item.get("estimated_fiber"))
    health_level = normalize_text(item.get("health_level")).lower()
    tags = normalize_text(item.get("tags")).lower()

    goals: List[str] = []

    if protein >= 25:
        goals.append("high_protein")

    if protein >= 30:
        goals.append("muscle_gain")

    if calories <= 550:
        goals.append("fat_loss")
        goals.append("low_calorie")

    if fiber >= 6:
        goals.append("high_fiber")

    if health_level in ["healthy", "very healthy"] or "healthy" in tags:
        goals.append("healthy_eating")

    if protein >= 20 and calories <= 700:
        goals.append("post_workout")

    goals.append("maintenance")

    unique_goals: List[str] = []

    for goal in goals:
        if goal not in unique_goals:
            unique_goals.append(goal)

    return unique_goals


def build_structured_input(
    user_request: str,
    goal: str,
    item: Dict[str, Any],
) -> str:
    cuisine = clean_cuisine(
        normalize_text(
            item.get("restaurant_category")
        )
    )

    calories = normalize_number(item.get("estimated_calories"))
    protein = normalize_number(item.get("estimated_protein"))
    fiber = normalize_number(item.get("estimated_fiber"))

    return (
        f"Goal:\n"
        f"{goal}\n\n"
        f"Cuisine:\n"
        f"{cuisine}\n\n"
        f"Calories:\n"
        f"{calories:.0f} kcal\n\n"
        f"Protein:\n"
        f"{protein:.0f} g\n\n"
        f"Fiber:\n"
        f"{fiber:.0f} g\n\n"
        f"User Request:\n"
        f"{user_request}"
    )


def build_user_requests(item: Dict[str, Any]) -> List[Dict[str, str]]:
    cuisine = clean_cuisine(
        normalize_text(
            item.get("restaurant_category")
        )
    )

    goals = infer_goals(item)

    requests: List[Dict[str, str]] = []

    for goal in goals:
        templates = GOAL_TEMPLATES.get(goal, [])

        for template in templates:
            user_request = template.format(cuisine=cuisine)

            requests.append(
                {
                    "goal": goal,
                    "constraint": goal,
                    "user_request": user_request,
                    "structured_input": build_structured_input(
                        user_request=user_request,
                        goal=goal,
                        item=item,
                    ),
                }
            )

    random.shuffle(requests)

    return requests[:QUERIES_PER_MENU]


def build_teacher_prompt(
    item: Dict[str, Any],
    structured_input: str,
) -> str:
    restaurant_name = normalize_text(item.get("restaurant_name"))
    menu_name = normalize_text(item.get("menu_name"))
    restaurant_category = normalize_text(item.get("restaurant_category"))
    menu_category = normalize_text(item.get("menu_category"))
    calories = normalize_number(item.get("estimated_calories"))
    protein = normalize_number(item.get("estimated_protein"))
    fiber = normalize_number(item.get("estimated_fiber"))
    health_level = normalize_text(item.get("health_level"))
    tags = normalize_text(item.get("tags"))

    return f"""
User Request and Constraints:

{structured_input}

Selected Restaurant:

{restaurant_name}

Restaurant Category:

{restaurant_category}

Selected Menu:

{menu_name}

Menu Category:

{menu_category}

Nutrition Evidence:

Calories:
{calories:.0f} kcal

Protein:
{protein:.0f} g

Fiber:
{fiber:.0f} g

Health Level:
{health_level}

Tags:
{tags}

Task:

Generate a natural explanation for why the selected restaurant and menu match the user's request.

Remember:

The recommendation has already been selected.
Do not recommend another restaurant.
Do not modify the restaurant name.
Do not modify the menu name.
Do not invent nutrition values.
Only generate the explanation.
""".strip()


def generate_explanation(prompt: str) -> str:
    response = ollama.chat(
        model=TEACHER_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    explanation = response["message"]["content"].strip()

    return explanation


def build_output(
    item: Dict[str, Any],
    explanation: str,
) -> str:
    restaurant_name = normalize_text(item.get("restaurant_name"))
    menu_name = normalize_text(item.get("menu_name"))
    calories = normalize_number(item.get("estimated_calories"))
    protein = normalize_number(item.get("estimated_protein"))
    fiber = normalize_number(item.get("estimated_fiber"))

    return (
        f"Recommended Restaurant:\n"
        f"{restaurant_name}\n\n"
        f"Recommended Menu:\n"
        f"{menu_name}\n\n"
        f"Nutrition Summary:\n"
        f"Calories:\n"
        f"{calories:.0f} kcal\n\n"
        f"Protein:\n"
        f"{protein:.0f} g\n\n"
        f"Fiber:\n"
        f"{fiber:.0f} g\n\n"
        f"Recommended Reason:\n"
        f"{explanation}"
    )


def build_metadata(
    item: Dict[str, Any],
    goal: str,
    constraint: str,
) -> Dict[str, Any]:
    return {
        "restaurant_id": item.get("restaurant_id"),
        "restaurant": normalize_text(item.get("restaurant_name")),
        "restaurant_category": normalize_text(item.get("restaurant_category")),
        "menu_id": item.get("menu_id"),
        "menu": normalize_text(item.get("menu_name")),
        "menu_category": normalize_text(item.get("menu_category")),
        "goal": goal,
        "constraint": constraint,
        "teacher_model": TEACHER_MODEL_NAME,
        "dataset_version": DATASET_VERSION,
        "estimated_calories": normalize_number(item.get("estimated_calories")),
        "estimated_protein": normalize_number(item.get("estimated_protein")),
        "estimated_fiber": normalize_number(item.get("estimated_fiber")),
        "health_level": normalize_text(item.get("health_level")),
        "tags": normalize_text(item.get("tags")),
    }


def build_sample(
    item: Dict[str, Any],
    request_info: Dict[str, str],
    explanation: str,
) -> Dict[str, Any]:
    return {
        "instruction": "Recommend a fitness meal based on the user's nutrition request.",
        "input": request_info["structured_input"],
        "output": build_output(
            item=item,
            explanation=explanation,
        ),
        "metadata": build_metadata(
            item=item,
            goal=request_info["goal"],
            constraint=request_info["constraint"],
        ),
    }


def is_valid_sample(sample: Dict[str, Any]) -> bool:
    output = normalize_text(sample.get("output"))

    metadata = sample.get("metadata", {})

    restaurant = normalize_text(
        metadata.get("restaurant")
    )

    menu = normalize_text(
        metadata.get("menu")
    )

    if restaurant == "":
        return False

    if menu == "":
        return False

    if restaurant not in output:
        return False

    if menu not in output:
        return False

    if "Recommended Reason:" not in output:
        return False

    return True


def save_generation_metadata(
    total_candidates: int,
    total_samples: int,
    failed_samples: int,
) -> None:
    metadata = {
        "dataset_version": DATASET_VERSION,
        "teacher_model": TEACHER_MODEL_NAME,
        "max_menu_candidates": MAX_MENU_CANDIDATES,
        "queries_per_menu": QUERIES_PER_MENU,
        "max_samples": MAX_SAMPLES,
        "total_candidates": total_candidates,
        "total_samples": total_samples,
        "failed_samples": failed_samples,
        "output_file": str(OUTPUT_FILE),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


def generate_dataset() -> None:
    random.seed(RANDOM_SEED)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = load_candidates()

    print("=" * 60)
    print("Fitness Home LoRA Dataset Generator V2")
    print("=" * 60)
    print(f"Teacher Model       : {TEACHER_MODEL_NAME}")
    print(f"Dataset Version     : {DATASET_VERSION}")
    print(f"Menu Candidates     : {len(candidates)}")
    print(f"Queries Per Menu    : {QUERIES_PER_MENU}")
    print(f"Max Samples         : {MAX_SAMPLES}")
    print(f"Output File         : {OUTPUT_FILE}")
    print("=" * 60)

    total_samples = 0
    failed_samples = 0
    seen_keys: Set[str] = set()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        for candidate_index, item in enumerate(candidates):
            if total_samples >= MAX_SAMPLES:
                break

            request_infos = build_user_requests(item)

            for request_info in request_infos:
                if total_samples >= MAX_SAMPLES:
                    break

                unique_key = (
                    f"{item.get('restaurant_id')}_"
                    f"{item.get('menu_id')}_"
                    f"{request_info['structured_input']}"
                )

                if unique_key in seen_keys:
                    continue

                seen_keys.add(unique_key)

                print(
                    f"[{total_samples + 1}/{MAX_SAMPLES}] "
                    f"{normalize_text(item.get('restaurant_name'))} -> "
                    f"{normalize_text(item.get('menu_name'))} | "
                    f"{request_info['goal']}"
                )

                prompt = build_teacher_prompt(
                    item=item,
                    structured_input=request_info["structured_input"],
                )

                try:
                    explanation = generate_explanation(prompt)

                    sample = build_sample(
                        item=item,
                        request_info=request_info,
                        explanation=explanation,
                    )

                    if not is_valid_sample(sample):
                        failed_samples += 1
                        continue

                    file.write(
                        json.dumps(
                            sample,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_samples += 1

                    if REQUEST_DELAY_SECONDS > 0:
                        time.sleep(REQUEST_DELAY_SECONDS)

                except Exception as error:
                    failed_samples += 1

                    print(
                        f"Generation failed: {error}"
                    )

                    continue

    save_generation_metadata(
        total_candidates=len(candidates),
        total_samples=total_samples,
        failed_samples=failed_samples,
    )

    print()
    print("=" * 60)
    print("Dataset Generation Completed")
    print("=" * 60)
    print(f"Total Samples  : {total_samples}")
    print(f"Failed Samples : {failed_samples}")
    print(f"Output File    : {OUTPUT_FILE}")
    print(f"Metadata File  : {METADATA_FILE}")
    print("=" * 60)


def main() -> None:
    generate_dataset()


if __name__ == "__main__":
    main()