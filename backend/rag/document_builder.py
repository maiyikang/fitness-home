import csv
import json
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RESTAURANTS_FILE = DATA_DIR / "restaurants_clean.csv"
MENUS_FILE = DATA_DIR / "menus_clean.csv"
OUTPUT_FILE = DATA_DIR / "rag_documents.jsonl"

MAX_MENU_ITEMS_PER_RESTAURANT = 8


def read_csv(file_path: Path) -> List[Dict[str, str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def group_menus_by_restaurant(menus: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}

    for menu in menus:
        restaurant_id = menu.get("restaurant_id", "")
        if restaurant_id == "":
            continue

        grouped.setdefault(restaurant_id, []).append(menu)

    return grouped


def build_restaurant_document(
    restaurant: Dict[str, str],
    menus: List[Dict[str, str]],
) -> Dict[str, object]:
    restaurant_id = restaurant.get("id", "")
    name = restaurant.get("name", "")
    category = restaurant.get("category", "")
    cuisine_tags = restaurant.get("cuisine_tags", "")
    average_calories = restaurant.get("average_calories", "")
    average_protein = restaurant.get("average_protein", "")
    average_fiber = restaurant.get("average_fiber", "")
    health_score = restaurant.get("health_score", "")
    fitness_score = restaurant.get("fitness_score", "")
    price_range = restaurant.get("price_range", "")
    address = restaurant.get("full_address", "")

    selected_menus = menus[:MAX_MENU_ITEMS_PER_RESTAURANT]

    menu_lines: List[str] = []

    for item in selected_menus:
        menu_name = item.get("name", "")
        menu_category = item.get("category", "")
        calories = item.get("estimated_calories", "")
        protein = item.get("estimated_protein", "")
        fiber = item.get("estimated_fiber", "")
        health_level = item.get("health_level", "")
        tags = item.get("tags", "")

        menu_lines.append(
            f"- {menu_name} ({menu_category}): "
            f"{calories} kcal, {protein}g protein, {fiber}g fiber, "
            f"health level: {health_level}, tags: {tags}"
        )

    menu_text = "\n".join(menu_lines)

    text = (
        f"Restaurant name: {name}\n"
        f"Restaurant ID: {restaurant_id}\n"
        f"Category: {category}\n"
        f"Cuisine tags: {cuisine_tags}\n"
        f"Price range: {price_range}\n"
        f"Address: {address}\n"
        f"Average calories: {average_calories} kcal\n"
        f"Average protein: {average_protein} g\n"
        f"Average fiber: {average_fiber} g\n"
        f"Health score: {health_score}\n"
        f"Fitness score: {fitness_score}\n"
        f"Representative menu items:\n"
        f"{menu_text}\n"
        f"Use this restaurant as evidence for fitness meal recommendation when it matches "
        f"the user's nutrition goal, cuisine preference, calorie target, protein target, "
        f"fiber target, and budget constraint."
    )

    return {
        "id": restaurant_id,
        "type": "restaurant_rag_document",
        "restaurant_id": restaurant_id,
        "restaurant_name": name,
        "category": category,
        "cuisine_tags": cuisine_tags,
        "average_calories": average_calories,
        "average_protein": average_protein,
        "average_fiber": average_fiber,
        "health_score": health_score,
        "fitness_score": fitness_score,
        "menu_count_in_document": len(selected_menus),
        "text": text,
    }


def write_jsonl(file_path: Path, documents: List[Dict[str, object]]) -> None:
    with file_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def main() -> None:
    print("Starting RAG document building...")

    restaurants = read_csv(RESTAURANTS_FILE)
    menus = read_csv(MENUS_FILE)

    print(f"Loaded restaurants: {len(restaurants)}")
    print(f"Loaded menu items: {len(menus)}")

    menus_by_restaurant = group_menus_by_restaurant(menus)

    documents: List[Dict[str, object]] = []

    for restaurant in restaurants:
        restaurant_id = restaurant.get("id", "")
        restaurant_menus = menus_by_restaurant.get(restaurant_id, [])

        if len(restaurant_menus) == 0:
            continue

        document = build_restaurant_document(restaurant, restaurant_menus)
        documents.append(document)

    write_jsonl(OUTPUT_FILE, documents)

    print(f"Created RAG documents: {len(documents)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("RAG document building completed.")


if __name__ == "__main__":
    main()