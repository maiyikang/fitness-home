from __future__ import annotations

from typing import List


def infer_cuisine_tags(category: str, restaurant_name: str) -> List[str]:
    """
    Infer cuisine tags from restaurant category and name.
    """

    text = f"{category} {restaurant_name}".lower()

    tags: List[str] = []

    keyword_map = {
        "sushi": ["sushi", "japanese", "poke"],
        "burger": ["burger"],
        "pizza": ["pizza"],
        "chinese": ["chinese", "dim sum", "noodle"],
        "korean": ["korean"],
        "thai": ["thai"],
        "indian": ["indian"],
        "mexican": ["mexican", "taco", "burrito"],
        "salad": ["salad", "healthy", "bowl"],
        "coffee": ["coffee", "tea", "cafe"],
        "dessert": ["dessert", "cake", "ice cream"],
        "breakfast": ["breakfast", "brunch"],
        "seafood": ["seafood", "fish", "shrimp"],
        "vegetarian": ["vegetarian", "vegan"],
    }

    for tag, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    if not tags:
        tags.append("general")

    return sorted(set(tags))


def estimate_calories(menu_name: str, description: str, category: str) -> int:
    """
    Estimate calories based on menu information.
    """

    text = f"{menu_name} {description} {category}".lower()

    rules = [
        (["salad", "smoothie", "soup"], 350),
        (["sushi", "poke"], 520),
        (["sandwich", "wrap"], 650),
        (["burger"], 900),
        (["pizza"], 850),
        (["dessert", "cake", "ice cream"], 700),
        (["breakfast", "waffle", "pancake"], 750),
        (["taco", "burrito"], 780),
        (["rice", "noodle", "curry"], 760),
    ]

    for keywords, calories in rules:
        if any(keyword in text for keyword in keywords):
            return calories

    return 650


def estimate_protein(menu_name: str, description: str, category: str) -> int:
    """
    Estimate protein content.
    """

    text = f"{menu_name} {description} {category}".lower()

    if any(word in text for word in ["chicken", "beef", "steak", "salmon", "tuna", "fish"]):
        return 38

    if any(word in text for word in ["burger", "sandwich", "burrito"]):
        return 28

    if any(word in text for word in ["sushi", "salad", "poke"]):
        return 24

    if any(word in text for word in ["pizza", "rice", "noodle"]):
        return 20

    return 15


def estimate_fiber(menu_name: str, description: str, category: str) -> int:
    """
    Estimate fiber content.
    """

    text = f"{menu_name} {description} {category}".lower()

    if any(word in text for word in ["salad", "vegetable", "vegan", "bean"]):
        return 9

    if any(word in text for word in ["bowl", "wrap", "burrito"]):
        return 7

    if any(word in text for word in ["rice", "sandwich", "noodle"]):
        return 4

    return 2


def calculate_health_score(
    calories: int,
    protein: int,
    fiber: int,
) -> int:
    """
    Calculate health score (0-100).
    """

    score = 0

    if calories <= 600:
        score += 40
    elif calories <= 750:
        score += 25

    if protein >= 30:
        score += 35
    elif protein >= 20:
        score += 20

    if fiber >= 7:
        score += 25
    elif fiber >= 4:
        score += 15

    return min(score, 100)


def calculate_cheat_score(calories: int) -> int:
    """
    Calculate cheat score (0-100).
    """

    if calories >= 900:
        return 100

    if calories >= 800:
        return 80

    if calories >= 700:
        return 50

    return 10


def calculate_fitness_score(
    health_score: int,
    cheat_score: int,
) -> int:
    """
    Overall fitness score.
    """

    score = health_score - int(cheat_score * 0.25) + 20

    return max(0, min(score, 100))


def determine_health_level(
    calories: int,
    protein: int,
    fiber: int,
) -> str:
    """
    Healthy / Normal / Cheat.
    """

    if calories <= 600 and protein >= 25:
        return "healthy"

    if calories >= 850:
        return "cheat"

    if fiber >= 7:
        return "healthy"

    return "normal"


def build_menu_tags(
    calories: int,
    protein: int,
    fiber: int,
    cuisine_tags: List[str],
) -> List[str]:
    """
    Build menu tags.
    """

    tags: List[str] = []

    if calories <= 500:
        tags.append("low_calorie")

    if calories >= 800:
        tags.append("high_calorie")

    if protein >= 30:
        tags.append("high_protein")

    if fiber >= 7:
        tags.append("high_fiber")

    if calories <= 600 and protein >= 25:
        tags.append("fitness_friendly")

    if calories >= 850:
        tags.append("cheat_meal")

    tags.extend(cuisine_tags)

    return sorted(set(tags))