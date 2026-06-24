import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RAW_RESTAURANTS_FILE = DATA_DIR / "restaurants.csv"
RAW_MENUS_FILE = DATA_DIR / "restaurant-menus.csv"

CLEAN_RESTAURANTS_FILE = DATA_DIR / "restaurants_clean.csv"
CLEAN_MENUS_FILE = DATA_DIR / "menus_clean.csv"
DATABASE_FILE = DATA_DIR / "fitness_home_rag_v1.db"
SUMMARY_FILE = DATA_DIR / "processing_summary.json"

MAX_RESTAURANTS = 5000
MAX_MENUS_PER_RESTAURANT = 30


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("&amp;", "&")
    text = text.replace("\u2028", " ")
    text = text.replace("\u2029", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        text = clean_text(value)
        if text == "":
            return default
        return float(text)
    except ValueError:
        return default


def parse_int(value: Optional[str], default: int = 0) -> int:
    try:
        text = clean_text(value)
        if text == "":
            return default
        return int(float(text))
    except ValueError:
        return default


def parse_price(value: Optional[str]) -> float:
    text = clean_text(value)
    if text == "":
        return 0.0

    match = re.search(r"\d+(\.\d+)?", text)
    if match is None:
        return 0.0

    return float(match.group())


def infer_cuisine_tags(category: str, name: str) -> List[str]:
    text = f"{category} {name}".lower()
    tags: List[str] = []

    keyword_map = {
        "sushi": ["sushi", "japanese", "poke"],
        "burger": ["burger", "burgers"],
        "pizza": ["pizza"],
        "chinese": ["chinese", "dim sum", "noodle"],
        "mexican": ["mexican", "taco", "burrito"],
        "indian": ["indian", "curry"],
        "thai": ["thai"],
        "korean": ["korean", "bbq"],
        "coffee": ["coffee", "tea", "smoothie"],
        "salad": ["salad", "healthy", "bowl"],
        "sandwich": ["sandwich", "subs", "deli"],
        "breakfast": ["breakfast", "brunch"],
        "dessert": ["dessert", "ice cream", "bakery", "cake"],
        "seafood": ["seafood", "fish", "shrimp"],
        "chicken": ["chicken", "wings"],
        "vegetarian": ["vegetarian", "vegan"],
    }

    for tag, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    if not tags:
        tags.append("general")

    return sorted(set(tags))


def estimate_calories(menu_name: str, description: str, category: str) -> int:
    text = f"{menu_name} {description} {category}".lower()

    calorie_rules = [
        (["salad", "soup", "smoothie", "tea"], 350),
        (["sushi", "poke", "bowl"], 520),
        (["sandwich", "wrap", "sub"], 650),
        (["chicken", "grilled"], 680),
        (["burger", "fries"], 900),
        (["pizza"], 850),
        (["dessert", "cake", "ice cream"], 700),
        (["breakfast", "pancake", "waffle"], 750),
        (["burrito", "taco", "nachos"], 780),
        (["curry", "rice", "noodle"], 760),
    ]

    for keywords, calories in calorie_rules:
        if any(keyword in text for keyword in keywords):
            return calories

    return 650


def estimate_protein(menu_name: str, description: str, category: str) -> int:
    text = f"{menu_name} {description} {category}".lower()

    if any(keyword in text for keyword in ["chicken", "beef", "steak", "salmon", "tuna", "fish", "shrimp"]):
        return 38

    if any(keyword in text for keyword in ["burger", "burrito", "bowl", "sandwich", "wrap"]):
        return 28

    if any(keyword in text for keyword in ["salad", "sushi", "poke"]):
        return 24

    if any(keyword in text for keyword in ["pizza", "pasta", "noodle", "rice"]):
        return 20

    if any(keyword in text for keyword in ["dessert", "cake", "ice cream", "drink", "tea"]):
        return 8

    return 18


def estimate_fiber(menu_name: str, description: str, category: str) -> int:
    text = f"{menu_name} {description} {category}".lower()

    if any(keyword in text for keyword in ["salad", "vegetable", "vegan", "vegetarian", "bean"]):
        return 9

    if any(keyword in text for keyword in ["bowl", "wrap", "burrito"]):
        return 7

    if any(keyword in text for keyword in ["sandwich", "rice", "noodle"]):
        return 4

    if any(keyword in text for keyword in ["burger", "pizza", "dessert"]):
        return 2

    return 3


def build_menu_tags(calories: int, protein: int, fiber: int, category: str, name: str) -> List[str]:
    tags: List[str] = []

    if calories <= 500:
        tags.append("low_calorie")
    elif calories >= 800:
        tags.append("high_calorie")

    if protein >= 30:
        tags.append("high_protein")

    if fiber >= 7:
        tags.append("high_fiber")

    if calories <= 600 and protein >= 25:
        tags.append("fitness_friendly")

    if calories >= 800:
        tags.append("cheat_meal")

    tags.extend(infer_cuisine_tags(category, name))

    return sorted(set(tags))


def get_health_level(calories: int, protein: int, fiber: int) -> str:
    if calories <= 600 and protein >= 25:
        return "healthy"

    if calories >= 850:
        return "cheat"

    if fiber >= 7 and calories <= 700:
        return "healthy"

    return "normal"


def read_restaurants() -> Dict[str, Dict[str, str]]:
    restaurants: Dict[str, Dict[str, str]] = {}

    with RAW_RESTAURANTS_FILE.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            restaurant_id = clean_text(row.get("id"))
            name = clean_text(row.get("name"))

            if restaurant_id == "" or name == "":
                continue

            if restaurant_id in restaurants:
                continue

            restaurants[restaurant_id] = {
                "id": restaurant_id,
                "name": name,
                "category": clean_text(row.get("category")),
                "score": str(parse_float(row.get("score"))),
                "ratings": str(parse_int(row.get("ratings"))),
                "price_range": clean_text(row.get("price_range")),
                "full_address": clean_text(row.get("full_address")),
                "zip_code": clean_text(row.get("zip_code")),
                "lat": str(parse_float(row.get("lat"))),
                "lng": str(parse_float(row.get("lng"))),
            }

            if len(restaurants) >= MAX_RESTAURANTS:
                break

    return restaurants


def read_menus(valid_restaurant_ids: set[str]) -> List[Dict[str, str]]:
    menu_items: List[Dict[str, str]] = []
    menu_count_by_restaurant: Dict[str, int] = {}

    with RAW_MENUS_FILE.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            restaurant_id = clean_text(row.get("restaurant_id"))

            if restaurant_id not in valid_restaurant_ids:
                continue

            current_count = menu_count_by_restaurant.get(restaurant_id, 0)
            if current_count >= MAX_MENUS_PER_RESTAURANT:
                continue

            category = clean_text(row.get("category"))
            name = clean_text(row.get("name"))
            description = clean_text(row.get("description"))
            price = parse_price(row.get("price"))

            if name == "":
                continue

            calories = estimate_calories(name, description, category)
            protein = estimate_protein(name, description, category)
            fiber = estimate_fiber(name, description, category)
            health_level = get_health_level(calories, protein, fiber)
            tags = build_menu_tags(calories, protein, fiber, category, name)

            menu_items.append(
                {
                    "id": str(len(menu_items) + 1),
                    "restaurant_id": restaurant_id,
                    "category": category,
                    "name": name,
                    "description": description,
                    "price": str(price),
                    "estimated_calories": str(calories),
                    "estimated_protein": str(protein),
                    "estimated_fiber": str(fiber),
                    "health_level": health_level,
                    "tags": json.dumps(tags, ensure_ascii=False),
                }
            )

            menu_count_by_restaurant[restaurant_id] = current_count + 1

    return menu_items


def calculate_restaurant_features(
    restaurants: Dict[str, Dict[str, str]],
    menu_items: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    menus_by_restaurant: Dict[str, List[Dict[str, str]]] = {}

    for item in menu_items:
        restaurant_id = item["restaurant_id"]
        menus_by_restaurant.setdefault(restaurant_id, []).append(item)

    cleaned_restaurants: List[Dict[str, str]] = []

    for restaurant_id, restaurant in restaurants.items():
        menus = menus_by_restaurant.get(restaurant_id, [])

        if len(menus) == 0:
            continue

        calories_values = [parse_int(item["estimated_calories"]) for item in menus]
        protein_values = [parse_int(item["estimated_protein"]) for item in menus]
        fiber_values = [parse_int(item["estimated_fiber"]) for item in menus]

        average_calories = round(sum(calories_values) / len(calories_values))
        average_protein = round(sum(protein_values) / len(protein_values))
        average_fiber = round(sum(fiber_values) / len(fiber_values))

        low_calorie_count = sum(1 for value in calories_values if value <= 600)
        high_protein_count = sum(1 for value in protein_values if value >= 30)
        high_fiber_count = sum(1 for value in fiber_values if value >= 7)

        health_score = round(
            (low_calorie_count / len(menus)) * 40
            + (high_protein_count / len(menus)) * 35
            + (high_fiber_count / len(menus)) * 25
        )

        cheat_score = round(
            sum(1 for value in calories_values if value >= 800) / len(menus) * 100
        )

        fitness_score = max(0, min(100, health_score - round(cheat_score * 0.25) + 20))

        cuisine_tags = infer_cuisine_tags(restaurant["category"], restaurant["name"])

        cleaned_restaurants.append(
            {
                **restaurant,
                "menu_count": str(len(menus)),
                "average_calories": str(average_calories),
                "average_protein": str(average_protein),
                "average_fiber": str(average_fiber),
                "health_score": str(health_score),
                "cheat_score": str(cheat_score),
                "fitness_score": str(fitness_score),
                "cuisine_tags": json.dumps(cuisine_tags, ensure_ascii=False),
            }
        )

    return cleaned_restaurants


def write_csv(file_path: Path, rows: List[Dict[str, str]]) -> None:
    if len(rows) == 0:
        raise ValueError(f"No rows to write: {file_path}")

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_database(restaurants: List[Dict[str, str]], menu_items: List[Dict[str, str]]) -> None:
    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE restaurants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            score REAL,
            ratings INTEGER,
            price_range TEXT,
            full_address TEXT,
            zip_code TEXT,
            lat REAL,
            lng REAL,
            menu_count INTEGER,
            average_calories INTEGER,
            average_protein INTEGER,
            average_fiber INTEGER,
            health_score INTEGER,
            cheat_score INTEGER,
            fitness_score INTEGER,
            cuisine_tags TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE menu_items (
            id TEXT PRIMARY KEY,
            restaurant_id TEXT NOT NULL,
            category TEXT,
            name TEXT NOT NULL,
            description TEXT,
            price REAL,
            estimated_calories INTEGER,
            estimated_protein INTEGER,
            estimated_fiber INTEGER,
            health_level TEXT,
            tags TEXT,
            FOREIGN KEY (restaurant_id) REFERENCES restaurants(id)
        )
        """
    )

    for restaurant in restaurants:
        cursor.execute(
            """
            INSERT INTO restaurants VALUES (
                :id,
                :name,
                :category,
                :score,
                :ratings,
                :price_range,
                :full_address,
                :zip_code,
                :lat,
                :lng,
                :menu_count,
                :average_calories,
                :average_protein,
                :average_fiber,
                :health_score,
                :cheat_score,
                :fitness_score,
                :cuisine_tags
            )
            """,
            restaurant,
        )

    for item in menu_items:
        cursor.execute(
            """
            INSERT INTO menu_items VALUES (
                :id,
                :restaurant_id,
                :category,
                :name,
                :description,
                :price,
                :estimated_calories,
                :estimated_protein,
                :estimated_fiber,
                :health_level,
                :tags
            )
            """,
            item,
        )

    cursor.execute("CREATE INDEX idx_menu_items_restaurant_id ON menu_items(restaurant_id)")
    cursor.execute("CREATE INDEX idx_restaurants_fitness_score ON restaurants(fitness_score)")
    cursor.execute("CREATE INDEX idx_restaurants_average_calories ON restaurants(average_calories)")

    connection.commit()
    connection.close()


def write_summary(restaurants: List[Dict[str, str]], menu_items: List[Dict[str, str]]) -> None:
    summary = {
        "max_restaurants": MAX_RESTAURANTS,
        "max_menus_per_restaurant": MAX_MENUS_PER_RESTAURANT,
        "clean_restaurant_count": len(restaurants),
        "clean_menu_item_count": len(menu_items),
        "output_files": {
            "restaurants_clean": str(CLEAN_RESTAURANTS_FILE),
            "menus_clean": str(CLEAN_MENUS_FILE),
            "database": str(DATABASE_FILE),
        },
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    print("Starting Fitness Home data processing...")

    if not RAW_RESTAURANTS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RAW_RESTAURANTS_FILE}")

    if not RAW_MENUS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RAW_MENUS_FILE}")

    restaurants = read_restaurants()
    print(f"Loaded restaurants: {len(restaurants)}")

    menu_items = read_menus(set(restaurants.keys()))
    print(f"Loaded menu items: {len(menu_items)}")

    cleaned_restaurants = calculate_restaurant_features(restaurants, menu_items)
    valid_restaurant_ids = {restaurant["id"] for restaurant in cleaned_restaurants}
    cleaned_menu_items = [
        item for item in menu_items if item["restaurant_id"] in valid_restaurant_ids
    ]

    print(f"Clean restaurants: {len(cleaned_restaurants)}")
    print(f"Clean menu items: {len(cleaned_menu_items)}")

    write_csv(CLEAN_RESTAURANTS_FILE, cleaned_restaurants)
    write_csv(CLEAN_MENUS_FILE, cleaned_menu_items)
    build_database(cleaned_restaurants, cleaned_menu_items)
    write_summary(cleaned_restaurants, cleaned_menu_items)

    print("Data processing completed.")
    print(f"Created: {CLEAN_RESTAURANTS_FILE}")
    print(f"Created: {CLEAN_MENUS_FILE}")
    print(f"Created: {DATABASE_FILE}")
    print(f"Created: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()