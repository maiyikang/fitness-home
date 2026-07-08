from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.paths import DATA_DIR, DATABASE_FILE


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_FILE = DATA_DIR / "lora_training_data.jsonl"
METADATA_FILE = DATA_DIR / "lora_training_metadata.json"

DATASET_VERSION = "v4_constraint_aware_llama31"

MAX_MENU_CANDIDATES = int(os.getenv("MAX_MENU_CANDIDATES", "1"))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "5"))
QUERIES_PER_MENU = int(os.getenv("QUERIES_PER_MENU", "5"))

MAX_NEW_TOKENS = 220
TEMPERATURE = 0.35
TOP_P = 0.9
REQUEST_DELAY_SECONDS = 0.1
RANDOM_SEED = 42


SYSTEM_PROMPT = """
You are the Teacher Model for the Fitness Home recommendation system.

The restaurant and menu item have already been selected by the retrieval system.

You are not allowed to recommend another restaurant.
You are not allowed to recommend another menu item.
You are not allowed to change the selected restaurant.
You are not allowed to change the selected menu item.

Your only task is to explain the retrieved recommendation using the provided evidence.

The restaurant name, menu name, cuisine, calories, protein, fiber, health level, and tags are factual database evidence.
Never modify factual evidence.
Never invent nutrition values.
Never infer ingredients.
Never infer dietary labels.
Never infer that a meal is vegetarian, healthy, balanced, or fitness-friendly unless the provided evidence explicitly supports it.

You must respect the provided constraint analysis.
Do not change the match status.
Do not claim that a violated constraint is satisfied.
If the constraint analysis says Partial Match, explain both strengths and limitations.
If the constraint analysis says Weak Match, clearly explain the limitations.
If the evidence is insufficient, state that the evidence is insufficient instead of guessing.

Your explanation must be objective, concise, and evidence-based.
Output only the explanation text.
Do not use markdown.
""".strip()


GOAL_TEMPLATES: Dict[str, List[str]] = {
    "fat_loss": [
        "I want a lower-calorie {cuisine} meal that supports fat loss.",
        "Recommend a {cuisine} meal for fat loss with reasonable calories.",
        "I need a fitness-friendly {cuisine} meal that is not too heavy.",
    ],
    "muscle_gain": [
        "I want a high-protein {cuisine} meal for muscle gain.",
        "Recommend a {cuisine} meal that supports strength training.",
        "I need a protein-focused {cuisine} meal after a workout.",
    ],
    "maintenance": [
        "I want a {cuisine} meal for daily fitness maintenance.",
        "Recommend a {cuisine} meal for regular fitness maintenance.",
        "I need an everyday {cuisine} meal that supports my fitness routine.",
    ],
    "high_fiber": [
        "I want a high-fiber {cuisine} meal for better nutrition.",
        "Recommend a {cuisine} meal with good fiber content.",
        "I need a {cuisine} meal that includes meaningful fiber.",
    ],
    "high_protein": [
        "I want a high-protein {cuisine} meal under 600 calories.",
        "Recommend a protein-rich {cuisine} meal.",
        "I need a {cuisine} meal with strong protein content.",
    ],
    "healthy_eating": [
        "I want a healthy {cuisine} meal recommendation.",
        "Recommend a nutrition-conscious {cuisine} meal.",
        "I need a nutritious {cuisine} meal option.",
    ],
    "post_workout": [
        "I want a post-workout {cuisine} meal.",
        "Recommend a {cuisine} meal after training.",
        "I need a recovery-friendly {cuisine} meal.",
    ],
    "low_calorie": [
        "I want a low-calorie {cuisine} meal.",
        "Recommend a lighter {cuisine} meal option.",
        "I need a {cuisine} meal that is not too high in calories.",
    ],
}


def normalize_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_number(value: Optional[Any], default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
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


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


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
            restaurants.name IS NOT NULL
            AND menu_items.name IS NOT NULL
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

    return [dict(row) for row in rows]


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

    if health_level == "healthy" or "healthy" in tags:
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
    cuisine = clean_cuisine(normalize_text(item.get("restaurant_category")))
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
    cuisine = clean_cuisine(normalize_text(item.get("restaurant_category")))
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


def evaluate_constraints(
    item: Dict[str, Any],
    request_info: Dict[str, str],
) -> Dict[str, Any]:
    goal = request_info["goal"]
    user_request = request_info["user_request"].lower()

    calories = normalize_number(item.get("estimated_calories"))
    protein = normalize_number(item.get("estimated_protein"))
    fiber = normalize_number(item.get("estimated_fiber"))
    health_level = normalize_text(item.get("health_level")).lower()
    tags = normalize_text(item.get("tags")).lower()

    satisfied: List[str] = []
    partially_satisfied: List[str] = []
    violated: List[str] = []
    limitations: List[str] = []

    if "under 600" in user_request:
        if calories <= 600:
            satisfied.append("Calorie target under 600 kcal is satisfied.")
        else:
            violated.append(
                f"Calorie target under 600 kcal is not satisfied because the menu has {calories:.0f} kcal."
            )

    if goal in ["low_calorie", "fat_loss"]:
        if calories <= 550:
            satisfied.append("Lower-calorie preference is satisfied.")
        elif calories <= 700:
            partially_satisfied.append(
                "Calorie level is moderate but not clearly low."
            )
        else:
            violated.append(
                "Lower-calorie preference is not satisfied because the calorie level is high."
            )

    if goal in ["high_protein", "muscle_gain", "post_workout"]:
        if protein >= 35:
            satisfied.append("Strong protein requirement is satisfied.")
        elif protein >= 25:
            partially_satisfied.append(
                "Protein content is meaningful but not very high."
            )
        else:
            violated.append(
                "High-protein requirement is not fully satisfied."
            )

    if goal == "high_fiber":
        if fiber >= 8:
            satisfied.append("High-fiber requirement is satisfied.")
        elif fiber >= 5:
            partially_satisfied.append(
                "Fiber content is moderate but not clearly high."
            )
        else:
            violated.append(
                "High-fiber requirement is not satisfied."
            )

    if goal == "healthy_eating":
        if health_level == "healthy" or "healthy" in tags or "fitness_friendly" in tags:
            satisfied.append("Healthy eating evidence is supported by the provided health level or tags.")
        else:
            partially_satisfied.append(
                "The request asks for healthy eating, but the evidence does not explicitly label this menu as healthy."
            )

    if goal == "maintenance":
        if calories <= 700 and protein >= 15:
            satisfied.append("Maintenance goal is reasonably supported by moderate calories and protein.")
        else:
            partially_satisfied.append(
                "Maintenance goal is only partially supported by the available nutrition evidence."
            )

    dessert_keywords = [
        "dessert",
        "cake",
        "cupcake",
        "sundae",
        "ice cream",
        "milkshake",
        "brownie",
        "baklava",
        "cookie",
        "soda",
        "sweet",
    ]

    menu_name = normalize_text(item.get("menu_name")).lower()
    menu_category = normalize_text(item.get("menu_category")).lower()

    if any(keyword in menu_name or keyword in menu_category for keyword in dessert_keywords):
        limitations.append(
            "The menu appears to be dessert-like or sweet, so it should not be overstated as a healthy fitness meal unless the evidence strongly supports it."
        )

    if len(violated) == 0 and len(partially_satisfied) == 0:
        overall_match = "Full Match"
    elif len(satisfied) > 0 and len(violated) == 0:
        overall_match = "Partial Match"
    elif len(satisfied) > 0:
        overall_match = "Partial Match"
    else:
        overall_match = "Weak Match"

    return {
        "overall_match": overall_match,
        "satisfied_constraints": satisfied,
        "partially_satisfied_constraints": partially_satisfied,
        "violated_constraints": violated,
        "limitations": limitations,
    }


def format_constraint_analysis(analysis: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("Overall Match:")
    lines.append(str(analysis["overall_match"]))
    lines.append("")

    lines.append("Satisfied Constraints:")
    if len(analysis["satisfied_constraints"]) == 0:
        lines.append("- None explicitly satisfied.")
    else:
        for item in analysis["satisfied_constraints"]:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("Partially Satisfied Constraints:")
    if len(analysis["partially_satisfied_constraints"]) == 0:
        lines.append("- None.")
    else:
        for item in analysis["partially_satisfied_constraints"]:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("Violated Constraints:")
    if len(analysis["violated_constraints"]) == 0:
        lines.append("- None.")
    else:
        for item in analysis["violated_constraints"]:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("Limitations:")
    if len(analysis["limitations"]) == 0:
        lines.append("- None.")
    else:
        for item in analysis["limitations"]:
            lines.append(f"- {item}")

    return "\n".join(lines)


def load_teacher() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print("=" * 60)
    print("Loading Llama 3.1 Teacher Model")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()

    print("Teacher model loaded successfully.")
    print("=" * 60)

    return tokenizer, model
    def build_teacher_prompt(
    item: Dict[str, Any],
    structured_input: str,
    constraint_analysis: Dict[str, Any],
) -> str:

    restaurant_name = normalize_text(item.get("restaurant_name"))
    restaurant_category = normalize_text(item.get("restaurant_category"))
    menu_name = normalize_text(item.get("menu_name"))
    menu_category = normalize_text(item.get("menu_category"))

    calories = normalize_number(item.get("estimated_calories"))
    protein = normalize_number(item.get("estimated_protein"))
    fiber = normalize_number(item.get("estimated_fiber"))

    health_level = normalize_text(item.get("health_level"))
    tags = normalize_text(item.get("tags"))

    constraint_report = format_constraint_analysis(
        constraint_analysis
    )

    return f"""
User Request

{structured_input}

Retrieved Restaurant

{restaurant_name}

Restaurant Category

{restaurant_category}

Retrieved Menu

{menu_name}

Menu Category

{menu_category}

Nutrition Evidence

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

Constraint Evaluation

{constraint_report}

Teacher Task

The recommendation has already been selected.

You MUST explain the retrieved recommendation.

You MUST follow the provided constraint evaluation.

Never change the Overall Match.

Never claim a violated constraint is satisfied.

Never invent nutrition facts.

Never infer ingredients.

Never infer health labels.

Never infer vegetarian, balanced, healthy or fitness-friendly unless explicitly supported by the evidence.

If the recommendation is only a Partial Match,
explain both its advantages and limitations.

If the recommendation is a Weak Match,
clearly explain why.

Do not recommend another restaurant.

Do not recommend another menu.

Output explanation only.
""".strip()


def generate_explanation(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
) -> str:

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

        {
            "role": "user",
            "content": prompt,
        },

    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            temperature=TEMPERATURE,

            top_p=TOP_P,

            do_sample=True,

            repetition_penalty=1.10,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id,

        )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    explanation = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return explanation


def build_output(
    item: Dict[str, Any],
    explanation: str,
) -> str:

    return (

        f"Recommended Restaurant:\n"

        f"{normalize_text(item['restaurant_name'])}\n\n"

        f"Recommended Menu:\n"

        f"{normalize_text(item['menu_name'])}\n\n"

        f"Nutrition Summary:\n"

        f"Calories:\n"

        f"{normalize_number(item['estimated_calories']):.0f} kcal\n\n"

        f"Protein:\n"

        f"{normalize_number(item['estimated_protein']):.0f} g\n\n"

        f"Fiber:\n"

        f"{normalize_number(item['estimated_fiber']):.0f} g\n\n"

        f"Recommended Reason:\n"

        f"{explanation}"

    )


def build_metadata(
    item: Dict[str, Any],
    goal: str,
    constraint: str,
    constraint_analysis: Dict[str, Any],
) -> Dict[str, Any]:

    return {

        "restaurant_id": item["restaurant_id"],

        "restaurant": item["restaurant_name"],

        "restaurant_category": item["restaurant_category"],

        "menu_id": item["menu_id"],

        "menu": item["menu_name"],

        "menu_category": item["menu_category"],

        "goal": goal,

        "constraint": constraint,

        "overall_match": constraint_analysis["overall_match"],

        "teacher_model": MODEL_NAME,

        "dataset_version": DATASET_VERSION,

        "estimated_calories": normalize_number(
            item["estimated_calories"]
        ),

        "estimated_protein": normalize_number(
            item["estimated_protein"]
        ),

        "estimated_fiber": normalize_number(
            item["estimated_fiber"]
        ),

        "health_level": normalize_text(
            item["health_level"]
        ),

        "tags": normalize_text(
            item["tags"]
        ),

    }
    def build_sample(
    item: Dict[str, Any],
    request_info: Dict[str, str],
    explanation: str,
    constraint_analysis: Dict[str, Any],
) -> Dict[str, Any]:

    return {

        "instruction":
            "Recommend a fitness meal based on the user's nutrition request.",

        "input":
            request_info["structured_input"],

        "output":
            build_output(
                item=item,
                explanation=explanation,
            ),

        "metadata":
            build_metadata(
                item=item,
                goal=request_info["goal"],
                constraint=request_info["constraint"],
                constraint_analysis=constraint_analysis,
            ),
    }


def is_valid_sample(
    sample: Dict[str, Any],
) -> bool:

    output = normalize_text(
        sample.get("output")
    )

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

        "teacher_model": MODEL_NAME,

        "max_menu_candidates": MAX_MENU_CANDIDATES,

        "queries_per_menu": QUERIES_PER_MENU,

        "max_samples": MAX_SAMPLES,

        "total_candidates": total_candidates,

        "generated_samples": total_samples,

        "failed_samples": failed_samples,

    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )


def generate_dataset() -> None:

    random.seed(RANDOM_SEED)

    tokenizer, model = load_teacher()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = load_candidates()

    print("=" * 60)
    print("Fitness Home Dataset Generator")
    print("=" * 60)
    print(f"Teacher Model : {MODEL_NAME}")
    print(f"Candidates    : {len(candidates)}")
    print(f"Max Samples   : {MAX_SAMPLES}")
    print("=" * 60)

    generated_samples = 0

    failed_samples = 0

    seen: Set[str] = set()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for item in candidates:

            if generated_samples >= MAX_SAMPLES:
                break

            request_infos = build_user_requests(
                item
            )

            for request_info in request_infos:

                if generated_samples >= MAX_SAMPLES:
                    break

                unique_key = (

                    f"{item['restaurant_id']}_"

                    f"{item['menu_id']}_"

                    f"{request_info['structured_input']}"

                )

                if unique_key in seen:
                    continue

                seen.add(unique_key)

                constraint_analysis = evaluate_constraints(
                    item=item,
                    request_info=request_info,
                )

                print(
                    f"[{generated_samples + 1}/{MAX_SAMPLES}] "
                    f"{item['restaurant_name']} -> "
                    f"{item['menu_name']} | "
                    f"{constraint_analysis['overall_match']}"
                )

                prompt = build_teacher_prompt(
                    item=item,
                    structured_input=request_info["structured_input"],
                    constraint_analysis=constraint_analysis,
                )

                try:

                    explanation = generate_explanation(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt,
                    )

                    sample = build_sample(
                        item=item,
                        request_info=request_info,
                        explanation=explanation,
                        constraint_analysis=constraint_analysis,
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

                    file.flush()

                    generated_samples += 1

                    if REQUEST_DELAY_SECONDS > 0:

                        time.sleep(
                            REQUEST_DELAY_SECONDS
                        )

                except Exception as error:

                    failed_samples += 1

                    print(
                        f"Generation failed: {error}"
                    )

                    continue

    save_generation_metadata(

        total_candidates=len(candidates),

        total_samples=generated_samples,

        failed_samples=failed_samples,

    )

    print()

    print("=" * 60)

    print("Dataset Generation Completed")

    print("=" * 60)

    print(f"Generated Samples : {generated_samples}")

    print(f"Failed Samples    : {failed_samples}")

    print(f"Output            : {OUTPUT_FILE}")

    print(f"Metadata          : {METADATA_FILE}")

    print("=" * 60)


def main() -> None:

    generate_dataset()


if __name__ == "__main__":

    main()