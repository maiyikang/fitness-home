#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    set_seed,
)

HERE = Path(__file__).resolve().parent

DEV_FILE = (
    HERE
    / "19_eval_protocol"
    / "development_benchmark_2069.jsonl"
)
M4_SCORED_FILE = (
    HERE
    / "41_explanation_baseline_eval"
    / "development_2069_frozen"
    / "m4_predictions_scored.jsonl"
)
STEP52_FILE = (
    HERE
    / "step52_run_explanation_baseline_m1_m5.py"
)
FILTER_FILE = (
    HERE.parent
    / "01_filter_v2"
    / "step14_filter_v2_3_calibration.py"
)

BENCHMARK_DIR = HERE / "45_vlm_multiconstraint_benchmark"
BENCHMARK_IMAGE_DIR = BENCHMARK_DIR / "images"
BENCHMARK_FILE = BENCHMARK_DIR / "vlm_benchmark_300.jsonl"
BENCHMARK_PROTOCOL_FILE = BENCHMARK_DIR / "vlm_benchmark_protocol.json"
BENCHMARK_SHA_FILE = BENCHMARK_DIR / "SHA256SUMS_BENCHMARK.txt"

RUN_ROOT = HERE / "46_vlm_multiconstraint_eval"

FOOD101_ROOT = Path(
    os.environ.get(
        "FITNESS_HOME_FOOD101_ROOT",
        str(HERE.parent / "data" / "food-101"),
    )
)
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

SEED = 20260818
BENCHMARK_SIZE = 300
SAMPLES_PER_CUISINE = 25
MAX_NEW_TOKENS = 220

METHOD_ORDER = ("T0", "V0", "V1", "V2", "V3")
METHOD_NAMES = {
    "T0": "Text-only RAG+LoRA TinyLlama (existing M4)",
    "V0": "Llama-3.2 Vision text-only + structured RAG",
    "V1": "Llama-3.2 Vision image-only without RAG",
    "V2": "Llama-3.2 Vision + image + structured RAG",
    "V3": (
        "Llama-3.2 Vision + image + structured RAG "
        "+ conflict-aware grounding policy"
    ),
}

CUISINE_TO_FOOD101_CLASS = {
    "American": "hamburger",
    "Chinese": "peking_duck",
    "Greek": "greek_salad",
    "Indian": "chicken_curry",
    "Italian": "pizza",
    "Japanese": "sushi",
    "Korean": "bibimbap",
    "Mediterranean": "hummus",
    "Mexican": "tacos",
    "Seafood": "fish_and_chips",
    "Thai": "pad_thai",
    "Vietnamese": "pho",
}
CUISINES = tuple(CUISINE_TO_FOOD101_CLASS)

SCENARIO_COUNTS_PER_CUISINE = {
    "aligned_clear": 10,
    "aligned_noisy": 5,
    "conflict_clear": 10,
}

CUISINE_ALIASES = {
    "American": (
        "american",
        "hamburger",
        "burger",
        "hot dog",
        "steak",
    ),
    "Chinese": (
        "chinese",
        "peking duck",
        "fried rice",
        "dumpling",
        "spring roll",
    ),
    "Greek": (
        "greek",
        "greek salad",
    ),
    "Indian": (
        "indian",
        "curry",
        "samosa",
    ),
    "Italian": (
        "italian",
        "pizza",
        "pasta",
        "spaghetti",
        "lasagna",
    ),
    "Japanese": (
        "japanese",
        "sushi",
        "ramen",
        "tempura",
    ),
    "Korean": (
        "korean",
        "bibimbap",
        "kimchi",
    ),
    "Mediterranean": (
        "mediterranean",
        "hummus",
        "falafel",
    ),
    "Mexican": (
        "mexican",
        "taco",
        "tacos",
        "burrito",
        "quesadilla",
    ),
    "Seafood": (
        "seafood",
        "fish",
        "oyster",
        "mussel",
        "lobster",
    ),
    "Thai": (
        "thai",
        "pad thai",
    ),
    "Vietnamese": (
        "vietnamese",
        "pho",
    ),
}

JSON_FIELDS = (
    "visual_cuisine_guess",
    "visual_quality",
    "visual_text_relation",
    "database_priority",
    "explanation",
)

SYSTEM_PROMPT = (
    "You are the Fitness Home multimodal recommendation evaluator. "
    "Return valid JSON only, with exactly these fields: "
    "visual_cuisine_guess, visual_quality, visual_text_relation, "
    "database_priority, explanation. "
    "The cuisine guess must be one of: "
    + ", ".join(CUISINES)
    + ", or uncertain. "
    "visual_quality must be clear, noisy, uncertain, or not_applicable. "
    "visual_text_relation must be aligned, conflict, uncertain, "
    "or not_applicable. "
    "database_priority must be true or false. "
    "The explanation must be one concise paragraph."
)

TEXT_ONLY_RAG_INSTRUCTION = (
    "No image is provided. Use only the user request and structured "
    "restaurant evidence. Set visual_cuisine_guess, visual_quality, and "
    "visual_text_relation to not_applicable. Set database_priority to true. "
    "Do not invent facts or alter numbers."
)

IMAGE_ONLY_INSTRUCTION = (
    "Use the image and user request only. No structured restaurant or "
    "nutrition evidence is provided. Do not invent calories, protein, fibre, "
    "or restaurant facts. If those facts are unavailable, say so clearly in "
    "the explanation. Set database_priority to false."
)

IMAGE_RAG_GENERIC_INSTRUCTION = (
    "Use the food image, user request, and structured restaurant evidence "
    "together to produce the JSON response."
)

IMAGE_RAG_POLICY_INSTRUCTION = (
    "Use the food image as complementary visual evidence only. "
    "The structured database evidence is authoritative for the restaurant "
    "identity, cuisine tags, calorie, protein, fibre, and constraint states. "
    "If the image is noisy, ambiguous, irrelevant, or conflicts with the "
    "text/database, explicitly mark the relation as conflict or uncertain "
    "and do not let the image override database facts. "
    "Set database_priority to true."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frozen 300-sample Food-101 + Fitness Home multimodal "
            "benchmark and run the V0-V3 Llama-3.2 Vision experiment."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test limit. Full run uses all 300 benchmark samples.",
    )
    parser.add_argument(
        "--methods",
        default="T0,V0,V1,V2,V3",
        help="Comma-separated subset of T0,V0,V1,V2,V3.",
    )
    parser.add_argument(
        "--overwrite-benchmark",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build/freeze benchmark but do not load the VLM.",
    )
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_of(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata", {})
    return value if isinstance(value, dict) else {}


def checks_of(record: dict[str, Any]) -> dict[str, bool]:
    raw = metadata_of(record).get("constraint_checks", {})
    if not isinstance(raw, dict):
        return {}
    return {
        ("fiber" if key == "fibre" else str(key)): bool(value)
        for key, value in raw.items()
    }


def strip_generation_instructions(text: str) -> str:
    value = str(text or "").strip()
    for marker in (
        "\n\nWrite one evidence-grounded recommendation explanation.",
        "\n\nTeacher-v4 reminder:",
    ):
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    return value


def balanced_record_selection(
    records: Sequence[dict[str, Any]],
    cuisine: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        record
        for record in records
        if str(
            metadata_of(record)
            .get("constraints", {})
            .get("cuisine", "")
        )
        == cuisine
        and checks_of(record).get("cuisine") is True
    ]

    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} cuisine-satisfied Development records "
            f"are available for {cuisine}; need {count}."
        )

    rng = random.Random(seed)
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        by_match[
            str(metadata_of(record).get("match_type", "")).lower()
        ].append(record)

    for values in by_match.values():
        rng.shuffle(values)

    order = ("full", "weak", "partial")
    selected: list[dict[str, Any]] = []

    while len(selected) < count:
        progress = False
        for match in order:
            if by_match[match]:
                selected.append(by_match[match].pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            break

    if len(selected) != count:
        raise RuntimeError(
            f"Balanced selection failed for {cuisine}: {len(selected)}"
        )

    return selected


def load_food101() -> Any:
    try:
        from torchvision.datasets import Food101
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required. Install a version compatible with "
            "the existing PyTorch build before running this script."
        ) from exc

    return Food101(
        root=str(FOOD101_ROOT),
        split="test",
        download=True,
    )


def food101_indices_by_class(dataset: Any) -> dict[str, list[int]]:
    classes = list(dataset.classes)
    labels = list(getattr(dataset, "_labels", []))
    if not labels:
        raise RuntimeError(
            "Food101 dataset does not expose the expected _labels field."
        )

    mapping: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        mapping[classes[int(label)]].append(index)
    return mapping


def add_noise(image: Image.Image) -> Image.Image:
    noisy = image.convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=8)
    )
    draw = ImageDraw.Draw(noisy)
    width, height = noisy.size
    left = int(width * 0.32)
    top = int(height * 0.32)
    right = int(width * 0.68)
    bottom = int(height * 0.68)
    draw.rectangle(
        (left, top, right, bottom),
        fill=(128, 128, 128),
    )
    return noisy


def save_selected_image(
    dataset: Any,
    dataset_index: int,
    output_path: Path,
    noisy: bool,
) -> None:
    image, _ = dataset[dataset_index]
    image = image.convert("RGB")
    if noisy:
        image = add_noise(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output_path,
        format="JPEG",
        quality=92,
        optimize=True,
    )


def benchmark_checksums() -> dict[str, str]:
    if not BENCHMARK_FILE.exists():
        return {}
    values = {
        BENCHMARK_FILE.name: sha256_file(BENCHMARK_FILE),
        BENCHMARK_PROTOCOL_FILE.name: sha256_file(
            BENCHMARK_PROTOCOL_FILE
        ),
    }
    for image_path in sorted(BENCHMARK_IMAGE_DIR.glob("*.jpg")):
        values[str(image_path.relative_to(BENCHMARK_DIR))] = (
            sha256_file(image_path)
        )
    return values


def write_benchmark_checksums() -> None:
    values = benchmark_checksums()
    with BENCHMARK_SHA_FILE.open("w", encoding="utf-8") as file:
        for name, digest in values.items():
            file.write(f"{digest}  {name}\n")


def verify_benchmark() -> None:
    if not (
        BENCHMARK_FILE.exists()
        and BENCHMARK_PROTOCOL_FILE.exists()
        and BENCHMARK_SHA_FILE.exists()
    ):
        raise FileNotFoundError(
            "Frozen VLM benchmark files are incomplete."
        )

    expected: dict[str, str] = {}
    with BENCHMARK_SHA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            digest, name = line.rstrip("\n").split(maxsplit=1)
            expected[name] = digest

    observed = benchmark_checksums()
    if expected != observed:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatch = sorted(
            name
            for name in expected.keys() & observed.keys()
            if expected[name] != observed[name]
        )
        raise RuntimeError(
            "Frozen VLM benchmark checksum failure. "
            f"Missing={missing[:5]} Extra={extra[:5]} "
            f"Mismatch={mismatch[:5]}"
        )


def build_benchmark(overwrite: bool) -> None:
    if overwrite and BENCHMARK_DIR.exists():
        shutil.rmtree(BENCHMARK_DIR)

    if (
        BENCHMARK_FILE.exists()
        and BENCHMARK_PROTOCOL_FILE.exists()
        and BENCHMARK_SHA_FILE.exists()
    ):
        verify_benchmark()
        print("Frozen VLM benchmark already exists and passed SHA256.")
        return

    for path in (DEV_FILE, M4_SCORED_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    development_records = read_jsonl(DEV_FILE)
    m4_rows = read_jsonl(M4_SCORED_FILE)
    m4_by_id = {
        str(row["sample_id"]): row
        for row in m4_rows
    }

    if len(development_records) != 2069:
        raise RuntimeError(
            f"Expected 2069 Development records, got "
            f"{len(development_records)}."
        )

    dataset = load_food101()
    indices_by_class = food101_indices_by_class(dataset)

    for cuisine, class_name in CUISINE_TO_FOOD101_CLASS.items():
        if class_name not in indices_by_class:
            raise RuntimeError(
                f"Required Food-101 class is missing: "
                f"{cuisine} -> {class_name}"
            )
        if len(indices_by_class[class_name]) < SAMPLES_PER_CUISINE:
            raise RuntimeError(
                f"Food-101 class {class_name} has too few test images."
            )

    rng = random.Random(SEED)

    image_pools: dict[str, list[int]] = {}
    for cuisine, class_name in CUISINE_TO_FOOD101_CLASS.items():
        candidates = list(indices_by_class[class_name])
        rng.shuffle(candidates)
        image_pools[cuisine] = candidates[:SAMPLES_PER_CUISINE]

    benchmark_rows: list[dict[str, Any]] = []
    selected_source_ids: set[str] = set()
    benchmark_index = 0

    for cuisine_index, cuisine in enumerate(CUISINES):
        selected_records = balanced_record_selection(
            development_records,
            cuisine,
            SAMPLES_PER_CUISINE,
            seed=SEED + cuisine_index,
        )

        scenario_labels = (
            ["aligned_clear"]
            * SCENARIO_COUNTS_PER_CUISINE["aligned_clear"]
            + ["aligned_noisy"]
            * SCENARIO_COUNTS_PER_CUISINE["aligned_noisy"]
            + ["conflict_clear"]
            * SCENARIO_COUNTS_PER_CUISINE["conflict_clear"]
        )
        rng_for_cuisine = random.Random(
            SEED + 1000 + cuisine_index
        )
        rng_for_cuisine.shuffle(scenario_labels)

        own_aligned_indices = iter(
            image_pools[cuisine][
                : (
                    SCENARIO_COUNTS_PER_CUISINE["aligned_clear"]
                    + SCENARIO_COUNTS_PER_CUISINE["aligned_noisy"]
                )
            ]
        )

        conflict_source_cuisine = CUISINES[
            (cuisine_index + 1) % len(CUISINES)
        ]
        conflict_indices = iter(
            image_pools[conflict_source_cuisine][
                -SCENARIO_COUNTS_PER_CUISINE["conflict_clear"] :
            ]
        )

        for source_record, scenario in zip(
            selected_records,
            scenario_labels,
        ):
            sample_id = str(source_record["sample_id"])
            if sample_id in selected_source_ids:
                raise RuntimeError(
                    f"Duplicate source sample selected: {sample_id}"
                )
            selected_source_ids.add(sample_id)

            benchmark_index += 1
            benchmark_id = f"VLM{benchmark_index:04d}"

            if scenario == "conflict_clear":
                image_cuisine = conflict_source_cuisine
                image_class = CUISINE_TO_FOOD101_CLASS[
                    image_cuisine
                ]
                image_index = next(conflict_indices)
                expected_relation = "conflict"
                expected_quality = "clear"
                noisy = False
            else:
                image_cuisine = cuisine
                image_class = CUISINE_TO_FOOD101_CLASS[
                    image_cuisine
                ]
                image_index = next(own_aligned_indices)
                expected_relation = "aligned"
                expected_quality = (
                    "noisy"
                    if scenario == "aligned_noisy"
                    else "clear"
                )
                noisy = scenario == "aligned_noisy"

            image_relative = Path("images") / f"{benchmark_id}.jpg"
            image_path = BENCHMARK_DIR / image_relative
            save_selected_image(
                dataset,
                image_index,
                image_path,
                noisy=noisy,
            )

            metadata = metadata_of(source_record)
            constraints = metadata.get("constraints", {})

            m4_row = m4_by_id.get(sample_id)
            if m4_row is None:
                raise RuntimeError(
                    f"M4 prediction missing for {sample_id}"
                )

            benchmark_rows.append({
                "benchmark_id": benchmark_id,
                "source_sample_id": sample_id,
                "source_constraint_signature_id": metadata.get(
                    "constraint_signature_id"
                ),
                "target_cuisine": cuisine,
                "source_match_type": metadata.get("match_type"),
                "scenario": scenario,
                "expected_visual_relation": expected_relation,
                "expected_visual_quality": expected_quality,
                "image_cuisine": image_cuisine,
                "food101_class": image_class,
                "food101_test_index": image_index,
                "image_file": str(image_relative),
                "query": metadata.get("query"),
                "restaurant_name": metadata.get("restaurant_name"),
                "constraints": constraints,
                "constraint_checks": metadata.get(
                    "constraint_checks"
                ),
                "structured_evidence": strip_generation_instructions(
                    source_record["input"]
                ),
                "reference_explanation": source_record["output"],
                "m4_text_baseline_prediction": m4_row["prediction"],
                "source_record": source_record,
            })

    if len(benchmark_rows) != BENCHMARK_SIZE:
        raise RuntimeError(
            f"Expected {BENCHMARK_SIZE} benchmark samples, got "
            f"{len(benchmark_rows)}."
        )

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(BENCHMARK_FILE, benchmark_rows)

    scenario_counts = Counter(
        row["scenario"]
        for row in benchmark_rows
    )
    cuisine_counts = Counter(
        row["target_cuisine"]
        for row in benchmark_rows
    )
    match_counts = Counter(
        str(row["source_match_type"])
        for row in benchmark_rows
    )

    protocol = {
        "experiment": (
            "fitness_home_vlm_multiconstraint_benchmark_v1"
        ),
        "status": "frozen_before_vlm_inference",
        "seed": SEED,
        "benchmark_samples": len(benchmark_rows),
        "development_only": True,
        "blind_test_used": False,
        "source_development_file": str(DEV_FILE),
        "source_development_sha256": sha256_file(DEV_FILE),
        "source_m4_file": str(M4_SCORED_FILE),
        "source_m4_sha256": sha256_file(M4_SCORED_FILE),
        "image_dataset": {
            "name": "Food-101",
            "split": "test",
            "root": str(FOOD101_ROOT),
            "cuisine_to_class": CUISINE_TO_FOOD101_CLASS,
        },
        "scenario_counts": dict(scenario_counts),
        "cuisine_counts": dict(cuisine_counts),
        "match_counts": dict(match_counts),
        "methods": {
            method_id: METHOD_NAMES[method_id]
            for method_id in METHOD_ORDER
        },
        "primary_comparisons": [
            "V0 vs V2: effect of adding image input to the same 11B model",
            "V1 vs V2: effect of structured RAG grounding",
            "V2 vs V3: effect of explicit conflict-aware grounding policy",
            "T0 vs V3: text-only deployed method vs proposed VLM module",
        ],
        "primary_metrics": [
            "Visual Cuisine Accuracy",
            "Visual Relation Safety Accuracy",
            "Visual Quality Safety Accuracy",
            "Database Priority Accuracy",
            "Conflict Resolution Exact Accuracy",
            "All-Constraint Exact Accuracy",
            "Constraint-State Macro-F1",
            "Numeric Relation Accuracy",
            "Failed-Constraint Recall",
            "Faithfulness",
            "Hallucination Rate",
            "ROUGE-L F1",
        ],
        "important_note": (
            "Food images supply visual category cues only. Structured "
            "restaurant evidence remains the authoritative source for "
            "restaurant identity and numerical nutrition values."
        ),
    }
    BENCHMARK_PROTOCOL_FILE.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_benchmark_checksums()
    verify_benchmark()

    print("=" * 78)
    print("VLM BENCHMARK BUILT AND FROZEN")
    print("=" * 78)
    print("Samples          :", len(benchmark_rows))
    print("Scenarios        :", dict(scenario_counts))
    print("Cuisines         :", dict(cuisine_counts))
    print("Match types      :", dict(match_counts))
    print("Blind test used  : NO")
    print("Benchmark        :", BENCHMARK_FILE)
    print("Protocol         :", BENCHMARK_PROTOCOL_FILE)
    print("Checksums        :", BENCHMARK_SHA_FILE)


def build_prompt(
    method_id: str,
    benchmark_row: dict[str, Any],
) -> tuple[str, Image.Image | None]:
    query = str(benchmark_row["query"])
    restaurant_name = str(benchmark_row["restaurant_name"])
    structured_evidence = str(
        benchmark_row["structured_evidence"]
    )
    image_path = (
        BENCHMARK_DIR
        / str(benchmark_row["image_file"])
    )

    if method_id == "V0":
        user_text = (
            f"{TEXT_ONLY_RAG_INSTRUCTION}\n\n"
            f"User request:\n{query}\n\n"
            f"Structured restaurant evidence:\n"
            f"{structured_evidence}"
        )
        return user_text, None

    if method_id == "V1":
        user_text = (
            f"{IMAGE_ONLY_INSTRUCTION}\n\n"
            f"User request:\n{query}\n\n"
            f"Selected restaurant name:\n{restaurant_name}"
        )
    elif method_id == "V2":
        user_text = (
            f"{IMAGE_RAG_GENERIC_INSTRUCTION}\n\n"
            f"User request:\n{query}\n\n"
            f"Structured restaurant evidence:\n"
            f"{structured_evidence}"
        )
    elif method_id == "V3":
        user_text = (
            f"{IMAGE_RAG_POLICY_INSTRUCTION}\n\n"
            f"User request:\n{query}\n\n"
            f"Structured restaurant evidence:\n"
            f"{structured_evidence}"
        )
    else:
        raise ValueError(f"Unsupported VLM method: {method_id}")

    image = Image.open(image_path).convert("RGB")
    return user_text, image


def chat_text(
    processor: Any,
    user_text: str,
    has_image: bool,
) -> str:
    content: list[dict[str, Any]] = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def quantization_config() -> BitsAndBytesConfig:
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_vlm() -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config(),
        device_map={"": 0},
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return processor, model


def move_inputs(
    inputs: dict[str, torch.Tensor],
    model: Any,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    moved: dict[str, torch.Tensor] = {}

    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(
                device=device,
                dtype=dtype,
            )
        else:
            moved[key] = value.to(device)
    return moved


def clean_json_text(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        text = text[first : last + 1]
    return text.strip()


def parse_model_json(raw: str) -> dict[str, Any]:
    cleaned = clean_json_text(raw)
    parsed: dict[str, Any] | None = None
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            parsed = value
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        return {
            "json_parse_success": False,
            "visual_cuisine_guess": "uncertain",
            "visual_quality": "uncertain",
            "visual_text_relation": "uncertain",
            "database_priority": False,
            "explanation": str(raw or "").strip(),
        }

    result = {
        "json_parse_success": all(
            field in parsed
            for field in JSON_FIELDS
        ),
        "visual_cuisine_guess": str(
            parsed.get(
                "visual_cuisine_guess",
                "uncertain",
            )
        ).strip(),
        "visual_quality": str(
            parsed.get("visual_quality", "uncertain")
        ).strip().lower(),
        "visual_text_relation": str(
            parsed.get(
                "visual_text_relation",
                "uncertain",
            )
        ).strip().lower(),
        "database_priority": bool(
            parsed.get("database_priority", False)
        ),
        "explanation": str(
            parsed.get("explanation", "")
        ).strip(),
    }

    if not result["explanation"]:
        result["explanation"] = str(raw or "").strip()
        result["json_parse_success"] = False

    return result


def canonical_cuisine(value: Any) -> str | None:
    text = re.sub(
        r"[_-]+",
        " ",
        str(value or "").strip().lower(),
    )

    if text in ("", "uncertain", "unknown", "not applicable"):
        return None

    for cuisine, aliases in CUISINE_ALIASES.items():
        if any(alias in text for alias in aliases):
            return cuisine
    return None


def normalise_relation(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "conflict" in text or "mismatch" in text:
        return "conflict"
    if "align" in text or "match" in text:
        return "aligned"
    if "not_applicable" in text or "not applicable" in text:
        return "not_applicable"
    return "uncertain"


def normalise_quality(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "noisy" in text or "blur" in text or "obscur" in text:
        return "noisy"
    if "clear" in text:
        return "clear"
    if "not_applicable" in text or "not applicable" in text:
        return "not_applicable"
    return "uncertain"


def visual_metrics(
    method_id: str,
    parsed: dict[str, Any],
    benchmark_row: dict[str, Any],
    explanation_metrics: dict[str, Any],
) -> dict[str, Any]:
    if method_id in ("T0", "V0"):
        return {
            "visual_applicable": False,
            "visual_cuisine_accuracy": None,
            "visual_relation_safety_accuracy": None,
            "visual_quality_safety_accuracy": None,
            "database_priority_accuracy": (
                True if method_id == "V0" else None
            ),
            "conflict_resolution_exact": None,
        }

    predicted_cuisine = canonical_cuisine(
        parsed["visual_cuisine_guess"]
    )
    expected_image_cuisine = str(
        benchmark_row["image_cuisine"]
    )
    cuisine_correct = (
        predicted_cuisine == expected_image_cuisine
    )

    predicted_relation = normalise_relation(
        parsed["visual_text_relation"]
    )
    expected_relation = str(
        benchmark_row["expected_visual_relation"]
    )
    scenario = str(benchmark_row["scenario"])

    if scenario == "aligned_noisy":
        relation_safe = predicted_relation in (
            "aligned",
            "uncertain",
        )
    else:
        relation_safe = (
            predicted_relation == expected_relation
        )

    predicted_quality = normalise_quality(
        parsed["visual_quality"]
    )
    expected_quality = str(
        benchmark_row["expected_visual_quality"]
    )
    if expected_quality == "noisy":
        quality_safe = predicted_quality in (
            "noisy",
            "uncertain",
        )
    else:
        quality_safe = predicted_quality == "clear"

    if method_id in ("V2", "V3"):
        database_priority_accuracy = bool(
            parsed["database_priority"]
        )
    else:
        database_priority_accuracy = None

    if scenario == "conflict_clear":
        conflict_exact = bool(
            predicted_relation == "conflict"
            and (
                parsed["database_priority"]
                if method_id in ("V2", "V3")
                else True
            )
            and explanation_metrics[
                "faithfulness_pass"
            ]
            and explanation_metrics[
                "numeric_faithful"
            ]
            and not explanation_metrics[
                "unsupported_health_or_goal_claim"
            ]
        )
    else:
        conflict_exact = None

    return {
        "visual_applicable": True,
        "visual_cuisine_accuracy": cuisine_correct,
        "visual_relation_safety_accuracy": relation_safe,
        "visual_quality_safety_accuracy": quality_safe,
        "database_priority_accuracy": (
            database_priority_accuracy
        ),
        "conflict_resolution_exact": conflict_exact,
    }


def load_completed(
    path: Path,
    allowed_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        benchmark_id = str(row["benchmark_id"])
        if benchmark_id not in allowed_ids:
            raise RuntimeError(
                f"Unknown benchmark_id in {path}: {benchmark_id}"
            )
        if benchmark_id in by_id:
            raise RuntimeError(
                f"Duplicate benchmark_id in {path}: {benchmark_id}"
            )
        by_id[benchmark_id] = row
    return by_id


def generate_vlm_method(
    method_id: str,
    benchmark_rows: Sequence[dict[str, Any]],
    processor: Any,
    model: Any,
    output_path: Path,
) -> None:
    allowed_ids = {
        str(row["benchmark_id"])
        for row in benchmark_rows
    }
    existing = load_completed(output_path, allowed_ids)

    pending = [
        row
        for row in benchmark_rows
        if str(row["benchmark_id"]) not in existing
    ]
    if not pending:
        print(f"[{method_id}] Predictions complete; skipping.")
        return

    print(
        f"[{method_id}] Existing={len(existing)} "
        f"Pending={len(pending)}"
    )

    completed = len(existing)

    for benchmark_row in pending:
        benchmark_id = str(benchmark_row["benchmark_id"])
        user_text, image = build_prompt(
            method_id,
            benchmark_row,
        )
        prompt = chat_text(
            processor,
            user_text,
            has_image=image is not None,
        )

        if image is None:
            inputs = processor(
                text=prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
        else:
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )

        inputs = move_inputs(dict(inputs), model)
        input_width = int(
            inputs["input_ids"].shape[-1]
        )

        start = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        elapsed = time.time() - start

        new_tokens = generated[:, input_width:]
        raw_output = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_model_json(raw_output)

        append_jsonl(
            output_path,
            {
                "benchmark_id": benchmark_id,
                "source_sample_id": benchmark_row[
                    "source_sample_id"
                ],
                "method_id": method_id,
                "method_name": METHOD_NAMES[method_id],
                "scenario": benchmark_row["scenario"],
                "raw_output": raw_output,
                "parsed": parsed,
                "generation_seconds": elapsed,
            },
        )

        completed += 1
        print(
            f"[{method_id}] {completed:03d}/"
            f"{len(benchmark_rows):03d} {benchmark_id}",
            flush=True,
        )

        if image is not None:
            image.close()


def ensure_text_baseline(
    benchmark_rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = [
        {
            "benchmark_id": row["benchmark_id"],
            "source_sample_id": row["source_sample_id"],
            "method_id": "T0",
            "method_name": METHOD_NAMES["T0"],
            "scenario": row["scenario"],
            "raw_output": row[
                "m4_text_baseline_prediction"
            ],
            "parsed": {
                "json_parse_success": None,
                "visual_cuisine_guess": "not_applicable",
                "visual_quality": "not_applicable",
                "visual_text_relation": "not_applicable",
                "database_priority": True,
                "explanation": row[
                    "m4_text_baseline_prediction"
                ],
            },
            "generation_seconds": 0.0,
        }
        for row in benchmark_rows
    ]
    write_jsonl(output_path, rows)


def aggregate_visual(
    scored: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    visual_rows = [
        row
        for row in scored
        if row["visual_metrics"]["visual_applicable"]
    ]
    conflict_rows = [
        row
        for row in scored
        if row["visual_metrics"][
            "conflict_resolution_exact"
        ]
        is not None
    ]
    db_rows = [
        row
        for row in scored
        if row["visual_metrics"][
            "database_priority_accuracy"
        ]
        is not None
    ]

    def rate(
        values: Sequence[bool],
    ) -> float | None:
        if not values:
            return None
        return sum(bool(value) for value in values) / len(values)

    return {
        "visual_sample_count": len(visual_rows),
        "visual_cuisine_accuracy": rate([
            row["visual_metrics"][
                "visual_cuisine_accuracy"
            ]
            for row in visual_rows
        ]),
        "visual_relation_safety_accuracy": rate([
            row["visual_metrics"][
                "visual_relation_safety_accuracy"
            ]
            for row in visual_rows
        ]),
        "visual_quality_safety_accuracy": rate([
            row["visual_metrics"][
                "visual_quality_safety_accuracy"
            ]
            for row in visual_rows
        ]),
        "database_priority_accuracy": rate([
            row["visual_metrics"][
                "database_priority_accuracy"
            ]
            for row in db_rows
        ]),
        "conflict_sample_count": len(conflict_rows),
        "conflict_resolution_exact_accuracy": rate([
            row["visual_metrics"][
                "conflict_resolution_exact"
            ]
            for row in conflict_rows
        ]),
        "json_parse_success_rate": rate([
            bool(row["parsed"]["json_parse_success"])
            for row in scored
            if row["parsed"]["json_parse_success"]
            is not None
        ]),
    }


def combined_summary(
    scored: Sequence[dict[str, Any]],
    step52: Any,
) -> dict[str, Any]:
    explanation_rows = [
        {
            "sample_id": row["benchmark_id"],
            "generation_seconds_estimate": row[
                "generation_seconds"
            ],
            "metrics": row["explanation_metrics"],
        }
        for row in scored
    ]
    summary = dict(
        step52.aggregate_summary(explanation_rows)
    )
    summary.update(aggregate_visual(scored))
    return summary


def subgroup_summaries(
    scored: Sequence[dict[str, Any]],
    step52: Any,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[
            f"scenario:{row['scenario']}"
        ].append(row)
        groups[
            f"match:{row['source_match_type']}"
        ].append(row)
        groups[
            f"cuisine:{row['target_cuisine']}"
        ].append(row)
    return {
        group: combined_summary(rows, step52)
        for group, rows in sorted(groups.items())
    }


def score_method(
    method_id: str,
    predictions: Sequence[dict[str, Any]],
    benchmark_by_id: dict[str, dict[str, Any]],
    step52: Any,
    strict_filter: Any | None,
    output_path: Path,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []

    for prediction in predictions:
        benchmark_id = str(
            prediction["benchmark_id"]
        )
        benchmark_row = benchmark_by_id[
            benchmark_id
        ]
        source_record = benchmark_row[
            "source_record"
        ]
        parsed = prediction["parsed"]
        explanation = str(parsed["explanation"])

        explanation_metrics = (
            step52.analyse_prediction(
                explanation,
                str(
                    benchmark_row[
                        "reference_explanation"
                    ]
                ),
                source_record,
                strict_filter,
            )
        )
        v_metrics = visual_metrics(
            method_id,
            parsed,
            benchmark_row,
            explanation_metrics,
        )

        scored.append({
            **prediction,
            "target_cuisine": benchmark_row[
                "target_cuisine"
            ],
            "image_cuisine": benchmark_row[
                "image_cuisine"
            ],
            "expected_visual_relation": benchmark_row[
                "expected_visual_relation"
            ],
            "expected_visual_quality": benchmark_row[
                "expected_visual_quality"
            ],
            "source_match_type": benchmark_row[
                "source_match_type"
            ],
            "visual_metrics": v_metrics,
            "explanation_metrics": explanation_metrics,
        })

    write_jsonl(output_path, scored)
    return scored


def write_main_table(
    csv_path: Path,
    md_path: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    headers = [
        "Method",
        "Visual Cuisine",
        "Relation Safety",
        "Quality Safety",
        "DB Priority",
        "Conflict Exact",
        "All-Constraint Exact",
        "State Macro-F1",
        "Numeric Relation",
        "Failed Recall",
        "Faithfulness",
        "Hallucination",
        "ROUGE-L",
    ]

    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        if method_id not in summaries:
            continue
        summary = summaries[method_id]
        rows.append({
            "Method": (
                f"{method_id} {METHOD_NAMES[method_id]}"
            ),
            "Visual Cuisine": summary.get(
                "visual_cuisine_accuracy"
            ),
            "Relation Safety": summary.get(
                "visual_relation_safety_accuracy"
            ),
            "Quality Safety": summary.get(
                "visual_quality_safety_accuracy"
            ),
            "DB Priority": summary.get(
                "database_priority_accuracy"
            ),
            "Conflict Exact": summary.get(
                "conflict_resolution_exact_accuracy"
            ),
            "All-Constraint Exact": summary.get(
                "all_constraint_exact_accuracy"
            ),
            "State Macro-F1": summary.get(
                "constraint_state_macro_f1"
            ),
            "Numeric Relation": summary.get(
                "numeric_relation_accuracy"
            ),
            "Failed Recall": summary.get(
                "failed_constraint_recall"
            ),
            "Faithfulness": summary.get(
                "faithfulness_rate"
            ),
            "Hallucination": summary.get(
                "hallucination_rate"
            ),
            "ROUGE-L": summary.get(
                "mean_rouge_l_f1"
            ),
        })

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for header in headers:
            value = row[header]
            if header == "Method":
                values.append(str(value))
            elif value is None:
                values.append("N/A")
            elif header == "ROUGE-L":
                values.append(f"{float(value):.4f}")
            else:
                values.append(f"{float(value):.4f}")
        markdown.append(
            "| " + " | ".join(values) + " |"
        )

    md_path.write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


def exact_mcnemar(
    first: Sequence[bool],
    second: Sequence[bool],
) -> dict[str, Any]:
    first_only = sum(
        a and not b
        for a, b in zip(first, second)
    )
    second_only = sum(
        b and not a
        for a, b in zip(first, second)
    )
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(first_only, second_only)
        tail = sum(
            math.comb(discordant, index)
            for index in range(smaller + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_only_success": first_only,
        "second_only_success": second_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def pairwise_comparison(
    first_id: str,
    second_id: str,
    scored_by_method: dict[
        str,
        list[dict[str, Any]],
    ],
    step52: Any,
) -> dict[str, Any]:
    first_by_id = {
        row["benchmark_id"]: row
        for row in scored_by_method[first_id]
    }
    second_by_id = {
        row["benchmark_id"]: row
        for row in scored_by_method[second_id]
    }
    ids = sorted(first_by_id)
    if ids != sorted(second_by_id):
        raise RuntimeError(
            f"Paired IDs mismatch: {first_id} vs {second_id}"
        )

    first_exact = [
        bool(
            first_by_id[bid][
                "explanation_metrics"
            ]["all_constraint_exact"]
        )
        for bid in ids
    ]
    second_exact = [
        bool(
            second_by_id[bid][
                "explanation_metrics"
            ]["all_constraint_exact"]
        )
        for bid in ids
    ]
    first_faith = [
        bool(
            first_by_id[bid][
                "explanation_metrics"
            ]["faithfulness_pass"]
        )
        for bid in ids
    ]
    second_faith = [
        bool(
            second_by_id[bid][
                "explanation_metrics"
            ]["faithfulness_pass"]
        )
        for bid in ids
    ]
    first_rouge = [
        float(
            first_by_id[bid][
                "explanation_metrics"
            ]["rouge_l_f1"]
        )
        for bid in ids
    ]
    second_rouge = [
        float(
            second_by_id[bid][
                "explanation_metrics"
            ]["rouge_l_f1"]
        )
        for bid in ids
    ]

    result = {
        "comparison": f"{first_id}_vs_{second_id}",
        "direction": (
            f"All differences are {second_id} minus {first_id}."
        ),
        "all_constraint_exact": exact_mcnemar(
            first_exact,
            second_exact,
        ),
        "faithfulness": exact_mcnemar(
            first_faith,
            second_faith,
        ),
        "rouge_l": step52.paired_bootstrap_difference(
            first_rouge,
            second_rouge,
            seed=SEED,
            repetitions=2000,
        ),
    }

    if first_id in ("V1", "V2", "V3") and second_id in (
        "V1",
        "V2",
        "V3",
    ):
        relation_first = [
            bool(
                first_by_id[bid][
                    "visual_metrics"
                ]["visual_relation_safety_accuracy"]
            )
            for bid in ids
        ]
        relation_second = [
            bool(
                second_by_id[bid][
                    "visual_metrics"
                ]["visual_relation_safety_accuracy"]
            )
            for bid in ids
        ]
        result["visual_relation_safety"] = exact_mcnemar(
            relation_first,
            relation_second,
        )

    return result


def main() -> None:
    args = parse_args()

    requested_methods = [
        method.strip().upper()
        for method in args.methods.split(",")
        if method.strip()
    ]
    invalid = [
        method
        for method in requested_methods
        if method not in METHOD_NAMES
    ]
    if invalid:
        raise ValueError(
            f"Unknown methods: {invalid}"
        )
    requested_methods = list(
        dict.fromkeys(requested_methods)
    )

    for path in (
        DEV_FILE,
        M4_SCORED_FILE,
        STEP52_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    build_benchmark(
        overwrite=args.overwrite_benchmark
    )
    verify_benchmark()

    if args.build_only:
        return

    if any(
        method in requested_methods
        for method in ("V0", "V1", "V2", "V3")
    ) and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Llama-3.2 Vision inference."
        )

    benchmark_rows_all = read_jsonl(BENCHMARK_FILE)
    benchmark_rows = (
        benchmark_rows_all[: args.limit]
        if args.limit is not None
        else benchmark_rows_all
    )
    benchmark_by_id = {
        str(row["benchmark_id"]): row
        for row in benchmark_rows
    }

    run_name = (
        f"smoke_{len(benchmark_rows)}"
        if args.limit is not None
        else "development_300"
    )
    run_dir = RUN_ROOT / run_name

    if args.overwrite_run and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_protocol = {
        "experiment": (
            "fitness_home_vlm_multiconstraint_eval_v1"
        ),
        "status": "frozen_before_result_inspection",
        "model": MODEL_ID,
        "benchmark_file": str(BENCHMARK_FILE),
        "benchmark_sha256": sha256_file(BENCHMARK_FILE),
        "benchmark_samples": len(benchmark_rows),
        "methods": requested_methods,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "quantization": "4-bit NF4 double quantization",
        "development_only": True,
        "blind_test_used": False,
    }
    run_protocol_path = run_dir / "run_protocol.json"
    if run_protocol_path.exists():
        existing = read_json(run_protocol_path)
        if existing != run_protocol:
            raise RuntimeError(
                "Existing VLM run protocol differs. "
                "Use --overwrite-run to restart."
            )
    else:
        run_protocol_path.write_text(
            json.dumps(
                run_protocol,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    print("=" * 78)
    print("FITNESS HOME — VLM MULTI-CONSTRAINT EXPERIMENT")
    print("=" * 78)
    print("Benchmark samples :", len(benchmark_rows))
    print("Methods           :", requested_methods)
    print("Model             :", MODEL_ID)
    print("Food-101 root     :", FOOD101_ROOT)
    print("Development only  : YES")
    print("Blind test used   : NO")
    print("Output            :", run_dir)

    prediction_paths = {
        method_id: (
            run_dir
            / f"{method_id.lower()}_predictions.jsonl"
        )
        for method_id in requested_methods
    }

    if "T0" in requested_methods:
        ensure_text_baseline(
            benchmark_rows,
            prediction_paths["T0"],
        )
        print("[T0] Existing text baseline prepared.")

    vlm_methods = [
        method
        for method in requested_methods
        if method in ("V0", "V1", "V2", "V3")
    ]

    processor = None
    model = None

    if vlm_methods:
        processor, model = load_vlm()

        for method_id in vlm_methods:
            generate_vlm_method(
                method_id,
                benchmark_rows,
                processor,
                model,
                prediction_paths[method_id],
            )

        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()

    step52 = load_module(
        STEP52_FILE,
        "fh_step52_vlm_metrics",
    )
    strict_filter = (
        load_module(
            FILTER_FILE,
            "fh_filter_v23_vlm",
        )
        if FILTER_FILE.exists()
        else None
    )

    scored_by_method: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    summaries: dict[str, dict[str, Any]] = {}
    subgroups: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for method_id in requested_methods:
        predictions = read_jsonl(
            prediction_paths[method_id]
        )
        if len(predictions) != len(benchmark_rows):
            raise RuntimeError(
                f"{method_id} predictions incomplete: "
                f"{len(predictions)}/{len(benchmark_rows)}"
            )

        scored_path = (
            run_dir
            / f"{method_id.lower()}_predictions_scored.jsonl"
        )
        scored = score_method(
            method_id,
            predictions,
            benchmark_by_id,
            step52,
            strict_filter,
            scored_path,
        )
        scored_by_method[method_id] = scored
        summaries[method_id] = combined_summary(
            scored,
            step52,
        )
        subgroups[method_id] = subgroup_summaries(
            scored,
            step52,
        )

    main_csv = run_dir / "vlm_main_table.csv"
    main_md = run_dir / "vlm_main_table.md"
    write_main_table(
        main_csv,
        main_md,
        summaries,
    )

    subgroup_rows: list[dict[str, Any]] = []
    for method_id, method_groups in subgroups.items():
        for group, summary in method_groups.items():
            subgroup_rows.append({
                "method_id": method_id,
                "method_name": METHOD_NAMES[method_id],
                "group": group,
                **summary,
            })

    subgroup_csv = run_dir / "vlm_subgroup_table.csv"
    headers = sorted({
        key
        for row in subgroup_rows
        for key in row
    })
    with subgroup_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )
        writer.writeheader()
        writer.writerows(subgroup_rows)

    pairs = (
        ("V0", "V2"),
        ("V1", "V2"),
        ("V2", "V3"),
        ("T0", "V3"),
    )
    significance = []
    for first_id, second_id in pairs:
        if (
            first_id in scored_by_method
            and second_id in scored_by_method
        ):
            significance.append(
                pairwise_comparison(
                    first_id,
                    second_id,
                    scored_by_method,
                    step52,
                )
            )

    significance_path = (
        run_dir / "vlm_pairwise_significance.json"
    )
    significance_path.write_text(
        json.dumps(
            significance,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = run_dir / "vlm_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": (
                    "fitness_home_vlm_multiconstraint_eval_v1"
                ),
                "development_only": True,
                "blind_test_used": False,
                "benchmark_samples": len(benchmark_rows),
                "methods": summaries,
                "subgroups": subgroups,
                "important_note": (
                    "Food-101 image labels provide visual cuisine "
                    "ground truth. Restaurant nutrition remains grounded "
                    "in the frozen structured Development evidence."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_files = [
        run_protocol_path,
        main_csv,
        main_md,
        subgroup_csv,
        significance_path,
        summary_path,
        *[
            prediction_paths[method]
            for method in requested_methods
        ],
        *[
            run_dir
            / f"{method.lower()}_predictions_scored.jsonl"
            for method in requested_methods
        ],
    ]

    checksum_path = run_dir / "SHA256SUMS.txt"
    with checksum_path.open("w", encoding="utf-8") as file:
        for path in output_files:
            file.write(
                f"{sha256_file(path)}  {path.name}\n"
            )

    print()
    print("=" * 78)
    print("VLM MULTI-CONSTRAINT EXPERIMENT COMPLETE")
    print("=" * 78)

    for method_id in METHOD_ORDER:
        if method_id not in summaries:
            continue
        summary = summaries[method_id]
        visual = summary.get(
            "visual_cuisine_accuracy"
        )
        relation = summary.get(
            "visual_relation_safety_accuracy"
        )
        conflict = summary.get(
            "conflict_resolution_exact_accuracy"
        )

        print(
            f"{method_id} "
            f"VisualCuisine="
            f"{'N/A' if visual is None else f'{visual:.2%}'} "
            f"Relation="
            f"{'N/A' if relation is None else f'{relation:.2%}'} "
            f"ConflictExact="
            f"{'N/A' if conflict is None else f'{conflict:.2%}'} "
            f"Exact={summary['all_constraint_exact_accuracy']:.2%} "
            f"StateF1={summary['constraint_state_macro_f1']:.2%} "
            f"Numeric={summary['numeric_relation_accuracy']:.2%} "
            f"Faith={summary['faithfulness_rate']:.2%} "
            f"Hallu={summary['hallucination_rate']:.2%} "
            f"ROUGE-L={summary['mean_rouge_l_f1']:.4f}"
        )

    print("Main table       :", main_md)
    print("Subgroup table   :", subgroup_csv)
    print("Significance     :", significance_path)
    print("Summary          :", summary_path)
    print("Blind test used  : NO")


if __name__ == "__main__":
    main()
