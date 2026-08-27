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
import statistics
import sys
import time
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import set_seed

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]

BLIND_SIGNATURE_FILE = (
    HERE
    / "19_eval_protocol"
    / "reserved_blind_signatures_250.jsonl"
)
EVAL_PROTOCOL_FILE = (
    HERE
    / "19_eval_protocol"
    / "evaluation_protocol.json"
)
EVAL_PROTOCOL_SHA_FILE = (
    HERE
    / "19_eval_protocol"
    / "sha256sums.txt"
)
DEVELOPMENT_FILE = (
    HERE
    / "19_eval_protocol"
    / "development_benchmark_2069.jsonl"
)

MAIN_SPLIT_DIR = HERE / "04_main20k_split"

STEP52_FILE = HERE / "step52_run_explanation_baseline_m1_m5.py"
STEP54_FILE = HERE / "step54_run_retrieval_baseline_eval.py"
STEP55_FILE = HERE / "step55_run_single_hybrid_rrf_decision.py"

FINAL_RETRIEVER_DECISION = (
    HERE
    / "44_hybrid_retrieval_postprocess"
    / "development_7500_frozen"
    / "final_retriever_decision.json"
)
if not FINAL_RETRIEVER_DECISION.exists():
    FINAL_RETRIEVER_DECISION = (
        HERE
        / "44_hybrid_retrieval_postprocess"
        / "development_7500"
        / "final_retriever_decision.json"
    )

FULL_ADAPTER_DIR = (
    HERE
    / "05_main20k_qlora_100pct"
    / "full_run_frozen"
    / "final_adapter"
)

PROMPT_FILE = (
    HERE
    / "40_explanation_baseline_protocol"
    / "prompt_templates.json"
)

FILTER_FILE = (
    HERE.parent
    / "01_filter_v2"
    / "step14_filter_v2_3_calibration.py"
)

OUT_ROOT = HERE / "49_final_blind_end_to_end"
LOCK_FILE = OUT_ROOT / "FINAL_BLIND_COMPLETE.lock"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SEED = 20260819
TOP_K = 5
CANDIDATE_K = 50
RRF_K = 60
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 180
BOOTSTRAP_REPETITIONS = 2000

RETRIEVAL_METHOD_ORDER = (
    "B1_BM25",
    "B2_BM25_ConstraintRerank",
    "B3_DenseBGE",
    "B4_DenseBGE_ConstraintRerank",
    "B6_HybridRRF_ConstraintRerank",
    "B5_StructuredConstraintOracle",
)

RETRIEVAL_METHOD_NAMES = {
    "B1_BM25": "BM25@5",
    "B2_BM25_ConstraintRerank": "BM25@50 + Constraint Rerank@5",
    "B3_DenseBGE": "BGE + FAISS Dense@5",
    "B4_DenseBGE_ConstraintRerank": (
        "BGE + FAISS Dense@50 + Constraint Rerank@5"
    ),
    "B6_HybridRRF_ConstraintRerank": (
        "BM25@50 ∪ Dense@50 → RRF@50 → Constraint Rerank@5"
    ),
    "B5_StructuredConstraintOracle": (
        "Structured Constraint Filter@5 (oracle parsed constraints)"
    ),
}

FIXED_EXPLANATION_METHODS = (
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
)

FIXED_EXPLANATION_NAMES = {
    "M1": "Structured Rule/Template",
    "M2": "Base TinyLlama without RAG evidence",
    "M3": "Dense evidence + Base TinyLlama",
    "M4": "Final Hybrid RAG + LoRA TinyLlama (100%)",
    "M5": "Hybrid RAG + Llama-3.1-8B-Instruct",
}

QUERY_TEMPLATES = (
    (
        "Find a {cuisine} restaurant for {goal} with no more than "
        "{max_calories} calories and at least {min_protein} g protein"
        "{fiber_clause}."
    ),
    (
        "I need a {cuisine} meal suitable for {goal}; keep it at or below "
        "{max_calories} kcal, with at least {min_protein} g protein"
        "{fiber_clause}."
    ),
    (
        "Recommend a {cuisine} option for {goal} under {max_calories} kcal "
        "that provides no less than {min_protein} g protein"
        "{fiber_clause}."
    ),
    (
        "Search for a {cuisine} restaurant that fits {goal}, capped at "
        "{max_calories} kcal with a minimum of {min_protein} g protein"
        "{fiber_clause}."
    ),
)

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FIELD_PATTERNS = {
    "average_calories": (
        re.compile(
            r"(?:average|avg)[ _-]*calories[^0-9]*"
            r"(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"calories[^0-9]*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    "average_protein": (
        re.compile(
            r"(?:average|avg)[ _-]*protein[^0-9]*"
            r"(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"protein[^0-9]*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    "average_fiber": (
        re.compile(
            r"(?:average|avg)[ _-]*(?:fiber|fibre)[^0-9]*"
            r"(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:fiber|fibre)[^0-9]*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-time final blind end-to-end Fitness Home evaluation "
            "over 250 sealed constraint signatures."
        )
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate sealed inputs, adapters, and protocol without running retrieval or models.",
    )
    parser.add_argument(
        "--tinyllama-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=4,
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise RuntimeError(f"Expected a JSON list in {path}")
        return [dict(row) for row in value]

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Expected JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_json_or_jsonl(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                values[parts[1].strip()] = parts[0].strip()
    return values


def metadata_of(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata", {})
    return value if isinstance(value, dict) else {}


def normalise_constraints(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("Constraint record is not a dictionary.")

    def required_value(*keys: str) -> Any:
        for key in keys:
            if key in raw and raw[key] not in (None, ""):
                return raw[key]
        raise RuntimeError(
            f"Missing required constraint field {keys}; keys={list(raw)}"
        )

    def optional_value(*keys: str) -> Any:
        for key in keys:
            if key in raw and raw[key] not in (None, ""):
                return raw[key]
        return None

    fiber = optional_value(
        "min_fiber",
        "min_fibre",
        "fiber",
        "fibre",
    )

    return {
        "cuisine": str(
            required_value("cuisine", "cuisine_type")
        ).strip(),
        "max_calories": int(
            float(
                required_value(
                    "max_calories",
                    "calorie_limit",
                    "maximum_calories",
                )
            )
        ),
        "min_protein": int(
            float(
                required_value(
                    "min_protein",
                    "protein_minimum",
                    "minimum_protein",
                )
            )
        ),
        "min_fiber": (
            None
            if fiber is None
            else int(float(fiber))
        ),
        "goal": str(
            required_value(
                "goal",
                "fitness_goal",
            )
        ).strip(),
    }


def constraint_key(constraints: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(constraints["cuisine"]).strip().lower(),
        int(constraints["max_calories"]),
        int(constraints["min_protein"]),
        (
            None
            if constraints.get("min_fiber") is None
            else int(constraints["min_fiber"])
        ),
        str(constraints["goal"]).strip().lower(),
    )


def constraints_from_any_row(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("constraints"), dict):
        return normalise_constraints(row["constraints"])

    if isinstance(row.get("signature"), dict):
        signature = row["signature"]
        if isinstance(signature.get("constraints"), dict):
            return normalise_constraints(
                signature["constraints"]
            )
        return normalise_constraints(signature)

    metadata = metadata_of(row)
    if isinstance(metadata.get("constraints"), dict):
        return normalise_constraints(
            metadata["constraints"]
        )

    return normalise_constraints(row)


def render_query(
    constraints: dict[str, Any],
    template_index: int,
) -> str:
    fiber = constraints.get("min_fiber")
    fiber_clause = (
        ""
        if fiber is None
        else f", and at least {int(fiber)} g dietary fibre"
    )

    return QUERY_TEMPLATES[template_index].format(
        cuisine=constraints["cuisine"],
        goal=constraints["goal"],
        max_calories=constraints["max_calories"],
        min_protein=constraints["min_protein"],
        fiber_clause=fiber_clause,
    )


def build_blind_signature_rows(
    source_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(source_rows) != 250:
        raise RuntimeError(
            f"Expected 250 reserved blind signatures, found {len(source_rows)}."
        )

    results: list[dict[str, Any]] = []
    keys: set[tuple[Any, ...]] = set()
    ids: set[str] = set()

    for index, row in enumerate(source_rows, start=1):
        constraints = constraints_from_any_row(row)
        key = constraint_key(constraints)
        if key in keys:
            raise RuntimeError(
                f"Duplicate blind constraint signature at row {index}: {key}"
            )
        keys.add(key)

        signature_id = str(
            row.get(
                "constraint_signature_id",
                row.get(
                    "signature_id",
                    f"BLINDSIG{index:04d}",
                ),
            )
        ).strip()
        if not signature_id:
            signature_id = f"BLINDSIG{index:04d}"
        if signature_id in ids:
            signature_id = f"BLINDSIG{index:04d}"
        ids.add(signature_id)

        results.append({
            "blind_signature_index": index,
            "constraint_signature_id": signature_id,
            "constraints": constraints,
            "source_record": row,
        })

    return results


def split_constraint_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[Any, ...]] = set()
    for row in read_jsonl(path):
        try:
            keys.add(
                constraint_key(
                    constraints_from_any_row(row)
                )
            )
        except Exception:
            continue
    return keys


def verify_no_overlap(
    blind_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    blind_keys = {
        constraint_key(row["constraints"])
        for row in blind_rows
    }

    main_keys: set[tuple[Any, ...]] = set()
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    ):
        main_keys |= split_constraint_keys(
            MAIN_SPLIT_DIR / filename
        )

    development_keys = split_constraint_keys(
        DEVELOPMENT_FILE
    )

    blind_main_overlap = blind_keys & main_keys
    blind_development_overlap = blind_keys & development_keys

    if blind_main_overlap:
        raise RuntimeError(
            f"Blind/Main-20K overlap detected: {len(blind_main_overlap)}"
        )
    if blind_development_overlap:
        raise RuntimeError(
            "Blind/Development overlap detected: "
            f"{len(blind_development_overlap)}"
        )

    return {
        "blind_signature_count": len(blind_keys),
        "main20k_signature_count": len(main_keys),
        "development_signature_count": len(development_keys),
        "blind_main_overlap": 0,
        "blind_development_overlap": 0,
    }


def find_reduced_adapters(
    budget: str,
) -> list[Path]:
    frozen = sorted(
        HERE.glob(
            f"29_random_{budget}_seed_*/"
            "full_run_frozen/final_adapter"
        )
    )
    if len(frozen) == 3:
        return frozen

    live = sorted(
        HERE.glob(
            f"29_random_{budget}_seed_*/"
            "full_run/final_adapter"
        )
    )
    if len(live) == 3:
        return live

    raise RuntimeError(
        f"Expected 3 frozen/live {budget} adapters; "
        f"found frozen={len(frozen)} live={len(live)}"
    )


def adapter_label(
    prefix: str,
    adapter_dir: Path,
) -> str:
    match = re.search(
        r"seed_(\d+)",
        str(adapter_dir),
    )
    seed = (
        match.group(1)
        if match
        else hashlib.sha256(
            str(adapter_dir).encode("utf-8")
        ).hexdigest()[:8]
    )
    return f"{prefix}_{seed}"


def preflight() -> dict[str, Any]:
    required_paths = (
        BLIND_SIGNATURE_FILE,
        EVAL_PROTOCOL_FILE,
        DEVELOPMENT_FILE,
        STEP52_FILE,
        STEP54_FILE,
        STEP55_FILE,
        FINAL_RETRIEVER_DECISION,
        FULL_ADAPTER_DIR / "adapter_config.json",
        FULL_ADAPTER_DIR / "adapter_model.safetensors",
        PROMPT_FILE,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    decision = read_json(FINAL_RETRIEVER_DECISION)
    if (
        decision.get("recommended_final_retriever")
        != "B6_HybridRRF_ConstraintRerank"
    ):
        raise RuntimeError(
            "Final retriever decision is not the frozen B6 Hybrid RRF method."
        )

    blind_source = read_json_or_jsonl(
        BLIND_SIGNATURE_FILE
    )
    blind_rows = build_blind_signature_rows(
        blind_source
    )
    overlap = verify_no_overlap(blind_rows)

    adapters_10 = find_reduced_adapters("10pct")
    adapters_5 = find_reduced_adapters("5pct")

    expected_sha = parse_sha_file(
        EVAL_PROTOCOL_SHA_FILE
    )
    source_sha_verified = None
    if expected_sha:
        matching_names = [
            name
            for name in expected_sha
            if Path(name).name
            == BLIND_SIGNATURE_FILE.name
        ]
        if matching_names:
            source_sha_verified = (
                sha256_file(BLIND_SIGNATURE_FILE)
                == expected_sha[
                    matching_names[0]
                ]
            )
            if not source_sha_verified:
                raise RuntimeError(
                    "Reserved blind signature SHA256 does not match "
                    "the frozen evaluation protocol."
                )

    return {
        "blind_rows": blind_rows,
        "overlap": overlap,
        "adapters_10": adapters_10,
        "adapters_5": adapters_5,
        "blind_signature_sha256": sha256_file(
            BLIND_SIGNATURE_FILE
        ),
        "blind_source_sha_verified": source_sha_verified,
    }


def nested_sources(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    sources = [result]
    for key in (
        "database_record",
        "restaurant",
        "record",
        "metadata",
    ):
        value = result.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def first_value(
    result: dict[str, Any],
    keys: Sequence[str],
) -> Any:
    for source in nested_sources(result):
        for key in keys:
            if key in source and source[key] not in (
                None,
                "",
                [],
                {},
            ):
                return source[key]
    return None


def parse_numeric_from_result(
    result: dict[str, Any],
    field: str,
) -> float:
    key_options = {
        "average_calories": (
            "average_calories",
            "avg_calories",
            "mean_calories",
            "calories",
        ),
        "average_protein": (
            "average_protein",
            "avg_protein",
            "mean_protein",
            "protein",
        ),
        "average_fiber": (
            "average_fiber",
            "average_fibre",
            "avg_fiber",
            "avg_fibre",
            "mean_fiber",
            "fiber",
            "fibre",
        ),
    }[field]

    value = first_value(result, key_options)
    if value not in (None, ""):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    text = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
    )
    for pattern in FIELD_PATTERNS[field]:
        match = pattern.search(text)
        if match:
            return float(match.group(1))

    raise RuntimeError(
        f"Could not extract {field} from retrieval result. "
        f"Top-level keys={list(result.keys())}"
    )


def normalise_checks(
    result: dict[str, Any],
) -> dict[str, bool]:
    raw = first_value(
        result,
        ("constraint_checks",),
    )
    if not isinstance(raw, dict):
        raise RuntimeError(
            "Selected retrieval result has no constraint_checks."
        )
    return {
        ("fiber" if key == "fibre" else str(key)): bool(value)
        for key, value in raw.items()
    }


def number_string(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def restaurant_name_of(
    result: dict[str, Any],
    step54: Any,
    step10: Any,
) -> str:
    name = first_value(
        result,
        (
            "restaurant_name",
            "name",
            "Restaurant Name",
        ),
    )
    if name not in (None, ""):
        return str(name).strip()

    restaurant_id = str(
        step54.restaurant_id_of(
            result,
            step10,
        )
    )
    return f"Restaurant {restaurant_id}"


def category_of(result: dict[str, Any]) -> str:
    value = first_value(
        result,
        (
            "category",
            "categories",
            "restaurant_category",
        ),
    )
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value or "Not provided")


def cuisine_tags_of(result: dict[str, Any]) -> list[str]:
    value = first_value(
        result,
        (
            "cuisine_tags",
            "cuisines",
            "tags",
            "cuisine",
        ),
    )
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return [
            item.strip()
            for item in re.split(
                r"[,;|]",
                stripped,
            )
            if item.strip()
        ]
    return []


def reference_explanation(
    restaurant_name: str,
    constraints: dict[str, Any],
    checks: dict[str, bool],
    calories: float,
    protein: float,
    fiber: float,
    match_type: str,
) -> str:
    satisfied: list[str] = []
    failed: list[str] = []

    cuisine = constraints["cuisine"]
    if checks["cuisine"]:
        satisfied.append(
            f"The cuisine requirement for {cuisine} is satisfied."
        )
    else:
        failed.append(
            f"the cuisine requirement for {cuisine} is not satisfied."
        )

    cal = number_string(calories)
    max_cal = number_string(
        constraints["max_calories"]
    )
    if checks["calories"]:
        satisfied.append(
            f"Its average calorie value of {cal} kcal is within "
            f"the maximum of {max_cal} kcal."
        )
    else:
        failed.append(
            f"its average calorie value of {cal} kcal exceeds "
            f"the maximum of {max_cal} kcal."
        )

    pro = number_string(protein)
    min_pro = number_string(
        constraints["min_protein"]
    )
    if checks["protein"]:
        satisfied.append(
            f"Its average protein value of {pro} g meets "
            f"the minimum of {min_pro} g."
        )
    else:
        failed.append(
            f"its average protein value of {pro} g is below "
            f"the minimum of {min_pro} g."
        )

    if constraints.get("min_fiber") is not None:
        fib = number_string(fiber)
        min_fib = number_string(
            constraints["min_fiber"]
        )
        if checks["fiber"]:
            satisfied.append(
                f"Its average fibre value of {fib} g meets "
                f"the minimum of {min_fib} g."
            )
        else:
            failed.append(
                f"its average fibre value of {fib} g is below "
                f"the minimum of {min_fib} g."
            )

    match_type = str(match_type).lower()
    if match_type == "full":
        opening = (
            f"{restaurant_name} satisfies all stated requirements."
        )
    elif match_type == "weak":
        opening = (
            f"{restaurant_name} meets some of the stated requirements."
        )
    else:
        opening = f"{restaurant_name} is a partial match."

    parts = [opening, *satisfied]
    if failed:
        failure_text = " ".join(failed)
        failure_text = (
            failure_text[0].upper()
            + failure_text[1:]
        )
        parts.append(
            f"However, {failure_text}"
        )

    return re.sub(
        r"\s+",
        " ",
        " ".join(parts),
    ).strip()


def build_explanation_record(
    signature_row: dict[str, Any],
    canonical_query: str,
    selected_result: dict[str, Any],
    step54: Any,
    step10: Any,
) -> dict[str, Any]:
    constraints = signature_row["constraints"]
    checks = normalise_checks(selected_result)

    required_checks = (
        "cuisine",
        "calories",
        "protein",
    )
    for name in required_checks:
        if name not in checks:
            raise RuntimeError(
                f"Selected result is missing required check: {name}"
            )

    if constraints.get("min_fiber") is not None:
        if "fiber" not in checks:
            raise RuntimeError(
                "Selected result is missing required fiber check."
            )

    restaurant_id = str(
        step54.restaurant_id_of(
            selected_result,
            step10,
        )
    )
    restaurant_name = restaurant_name_of(
        selected_result,
        step54,
        step10,
    )
    category = category_of(selected_result)
    cuisine_tags = cuisine_tags_of(selected_result)

    calories = parse_numeric_from_result(
        selected_result,
        "average_calories",
    )
    protein = parse_numeric_from_result(
        selected_result,
        "average_protein",
    )
    fiber = parse_numeric_from_result(
        selected_result,
        "average_fiber",
    )

    match_type = str(
        first_value(
            selected_result,
            ("match_type",),
        )
        or "partial"
    ).lower()

    lines = [
        "User request:",
        canonical_query,
        "",
        "User constraints:",
        f"- Cuisine: {constraints['cuisine']}",
        (
            f"- Maximum calories: "
            f"{constraints['max_calories']} kcal"
        ),
        (
            f"- Minimum protein: "
            f"{constraints['min_protein']} g"
        ),
    ]
    if constraints.get("min_fiber") is not None:
        lines.append(
            f"- Minimum fibre: "
            f"{constraints['min_fiber']} g"
        )
    lines.extend([
        f"- Fitness goal: {constraints['goal']}",
        "",
        "Selected restaurant evidence:",
        f"- Restaurant name: {restaurant_name}",
        f"- Category: {category}",
        (
            "- Cuisine tags: "
            + json.dumps(
                cuisine_tags,
                ensure_ascii=False,
            )
        ),
        (
            f"- Average calories: "
            f"{number_string(calories)} kcal"
        ),
        (
            f"- Average protein: "
            f"{number_string(protein)} g"
        ),
        (
            f"- Average fibre: "
            f"{number_string(fiber)} g"
        ),
        "",
        "Constraint evaluation:",
        (
            "- Cuisine requirement: "
            + (
                "Satisfied"
                if checks["cuisine"]
                else "Not satisfied"
            )
        ),
        (
            "- Calorie limit: "
            + (
                "Satisfied"
                if checks["calories"]
                else "Not satisfied"
            )
        ),
        (
            "- Protein minimum: "
            + (
                "Satisfied"
                if checks["protein"]
                else "Not satisfied"
            )
        ),
    ])
    if constraints.get("min_fiber") is not None:
        lines.append(
            "- Fibre minimum: "
            + (
                "Satisfied"
                if checks["fiber"]
                else "Not satisfied"
            )
        )

    lines.extend([
        "",
        "Overall match:",
        match_type,
        "",
        "Write one evidence-grounded recommendation explanation.",
        "",
        "Do not add facts that are not shown above.",
    ])

    reference = reference_explanation(
        restaurant_name,
        constraints,
        checks,
        calories,
        protein,
        fiber,
        match_type,
    )

    sample_id = (
        f"BLIND_{signature_row['constraint_signature_id']}"
    )

    return {
        "sample_id": sample_id,
        "instruction": (
            "Generate an evidence-grounded restaurant "
            "recommendation explanation."
        ),
        "input": "\n".join(lines),
        "output": reference,
        "metadata": {
            "dataset_version": (
                "final_blind_end_to_end_v1"
            ),
            "query_id": (
                f"BLINDQ"
                f"{signature_row['blind_signature_index']:04d}"
            ),
            "query": canonical_query,
            "constraints": constraints,
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant_name,
            "match_type": match_type,
            "constraint_checks": checks,
            "constraint_signature_id": (
                signature_row[
                    "constraint_signature_id"
                ]
            ),
            "filter_v2_3_accepted": True,
            "accepted": True,
            "blind_test_record": True,
        },
    }


def load_generic_lora_model(
    adapter_dir: Path,
    step52: Any,
) -> Any:
    base = step52.load_base_model(BASE_MODEL)
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        is_trainable=False,
    )
    step52.configure_generation(model)
    return model


def ensure_template_predictions(
    records: Sequence[dict[str, Any]],
    output_path: Path,
    method_name: str,
) -> None:
    rows = [
        {
            "sample_id": record["sample_id"],
            "method_id": "M1",
            "method_name": method_name,
            "prediction": record["output"],
            "reference": record["output"],
            "user_content_sha256": None,
            "generation_seconds_estimate": 0.0,
        }
        for record in records
    ]
    write_jsonl(output_path, rows)


def aggregate_end_to_end(
    scored: Sequence[dict[str, Any]],
    retrieval_full_by_sample: dict[str, bool],
    step52: Any,
) -> dict[str, Any]:
    summary = dict(step52.aggregate_summary(scored))

    end_to_end_exact = []
    end_to_end_faithful_full = []

    for row in scored:
        sample_id = str(row["sample_id"])
        retrieval_full = bool(
            retrieval_full_by_sample[sample_id]
        )
        metrics = row["metrics"]
        end_to_end_exact.append(
            retrieval_full
            and bool(
                metrics["all_constraint_exact"]
            )
        )
        end_to_end_faithful_full.append(
            retrieval_full
            and bool(
                metrics["faithfulness_pass"]
            )
        )

    summary["retrieval_full_at_1_rate"] = (
        sum(
            retrieval_full_by_sample.values()
        )
        / len(retrieval_full_by_sample)
    )
    summary["end_to_end_exact_accuracy"] = (
        sum(end_to_end_exact)
        / len(end_to_end_exact)
    )
    summary[
        "end_to_end_full_and_faithful_rate"
    ] = (
        sum(end_to_end_faithful_full)
        / len(end_to_end_faithful_full)
    )
    return summary


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError(
            "Paired vectors have different lengths."
        )
    differences = [
        float(b) - float(a)
        for a, b in zip(first, second)
    ]
    observed = statistics.fmean(
        differences
    )

    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sample = [
            differences[
                rng.randrange(len(differences))
            ]
            for _ in range(len(differences))
        ]
        bootstrap_means.append(
            statistics.fmean(sample)
        )
    bootstrap_means.sort()

    return {
        "difference_second_minus_first": observed,
        "bootstrap_95ci": [
            bootstrap_means[
                int(
                    0.025
                    * (
                        BOOTSTRAP_REPETITIONS
                        - 1
                    )
                )
            ],
            bootstrap_means[
                int(
                    0.975
                    * (
                        BOOTSTRAP_REPETITIONS
                        - 1
                    )
                )
            ],
        ],
        "repetitions": BOOTSTRAP_REPETITIONS,
    }


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
        smaller = min(
            first_only,
            second_only,
        )
        tail = sum(
            math.comb(
                discordant,
                index,
            )
            for index in range(
                smaller + 1
            )
        ) / (2 ** discordant)
        p_value = min(
            1.0,
            2.0 * tail,
        )

    return {
        "first_only_success": first_only,
        "second_only_success": second_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def write_retrieval_table(
    path: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    headers = [
        "Method",
        "Precision@5",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Hit@5",
        "Top1 CSR",
        "Full@1",
        "Full@5",
        "Unique Top1",
        "Largest Top1 Share",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            ["---"] * len(headers)
        )
        + " |",
    ]

    for method_id in RETRIEVAL_METHOD_ORDER:
        summary = summaries[method_id]
        values = [
            (
                f"{method_id} "
                f"{RETRIEVAL_METHOD_NAMES[method_id]}"
            ),
            f"{summary['mean_precision_at_5']:.4f}",
            f"{summary['mean_recall_at_5']:.4f}",
            f"{summary['mrr']:.4f}",
            f"{summary['mean_ndcg_at_5']:.4f}",
            f"{summary['hit_at_5_rate']:.4f}",
            (
                f"{summary['mean_top1_constraint_satisfaction']:.4f}"
            ),
            f"{summary['full_match_at_1_rate']:.4f}",
            f"{summary['full_match_at_5_rate']:.4f}",
            str(
                summary[
                    "unique_top1_restaurants"
                ]
            ),
            f"{summary['largest_top1_share']:.4f}",
        ]
        lines.append(
            "| " + " | ".join(values) + " |"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_explanation_table(
    path: Path,
    summaries: OrderedDict[str, dict[str, Any]],
    names: dict[str, str],
) -> None:
    headers = [
        "Method",
        "All-Constraint Exact",
        "State Macro-F1",
        "Numeric Relation",
        "Failed Recall",
        "Faithfulness",
        "Hallucination",
        "ROUGE-L",
        "Retrieval Full@1",
        "End-to-End Exact",
        "Full+Faithful",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            ["---"] * len(headers)
        )
        + " |",
    ]

    for method_id, summary in summaries.items():
        values = [
            f"{method_id} {names[method_id]}",
            (
                f"{summary['all_constraint_exact_accuracy']:.4f}"
            ),
            (
                f"{summary['constraint_state_macro_f1']:.4f}"
            ),
            (
                f"{summary['numeric_relation_accuracy']:.4f}"
            ),
            (
                f"{summary['failed_constraint_recall']:.4f}"
            ),
            f"{summary['faithfulness_rate']:.4f}",
            f"{summary['hallucination_rate']:.4f}",
            f"{summary['mean_rouge_l_f1']:.4f}",
            (
                f"{summary['retrieval_full_at_1_rate']:.4f}"
            ),
            (
                f"{summary['end_to_end_exact_accuracy']:.4f}"
            ),
            (
                f"{summary['end_to_end_full_and_faithful_rate']:.4f}"
            ),
        ]
        lines.append(
            "| " + " | ".join(values) + " |"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def reduced_group_summary(
    method_ids: Sequence[str],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = (
        "all_constraint_exact_accuracy",
        "constraint_state_macro_f1",
        "numeric_relation_accuracy",
        "failed_constraint_recall",
        "faithfulness_rate",
        "hallucination_rate",
        "mean_rouge_l_f1",
        "end_to_end_exact_accuracy",
        "end_to_end_full_and_faithful_rate",
    )

    result: dict[str, Any] = {
        "model_count": len(method_ids),
        "methods": list(method_ids),
        "metrics": {},
    }

    for metric in metrics:
        values = [
            float(summaries[method_id][metric])
            for method_id in method_ids
        ]
        result["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "sample_sd": (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),
            "min": min(values),
            "max": max(values),
            "per_model": values,
        }
    return result


def write_reduced_table(
    path: Path,
    groups: dict[str, dict[str, Any]],
) -> None:
    headers = [
        "Budget",
        "Models",
        "Exact Mean±SD",
        "Faith Mean±SD",
        "ROUGE-L Mean±SD",
        "End-to-End Exact Mean±SD",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            ["---"] * len(headers)
        )
        + " |",
    ]

    for budget, group in groups.items():
        metrics = group["metrics"]
        values = [
            budget,
            str(group["model_count"]),
            (
                f"{metrics['all_constraint_exact_accuracy']['mean']:.4f}"
                f" ± "
                f"{metrics['all_constraint_exact_accuracy']['sample_sd']:.4f}"
            ),
            (
                f"{metrics['faithfulness_rate']['mean']:.4f}"
                f" ± "
                f"{metrics['faithfulness_rate']['sample_sd']:.4f}"
            ),
            (
                f"{metrics['mean_rouge_l_f1']['mean']:.4f}"
                f" ± "
                f"{metrics['mean_rouge_l_f1']['sample_sd']:.4f}"
            ),
            (
                f"{metrics['end_to_end_exact_accuracy']['mean']:.4f}"
                f" ± "
                f"{metrics['end_to_end_exact_accuracy']['sample_sd']:.4f}"
            ),
        ]
        lines.append(
            "| " + " | ".join(values) + " |"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if LOCK_FILE.exists():
        print(
            "FINAL BLIND TEST IS ALREADY COMPLETE AND LOCKED."
        )
        print("Lock:", LOCK_FILE)
        return

    preflight_data = preflight()
    blind_rows = preflight_data["blind_rows"]
    adapters_10 = preflight_data[
        "adapters_10"
    ]
    adapters_5 = preflight_data[
        "adapters_5"
    ]

    reduced_method_names: dict[str, str] = {}
    adapter_method_dirs: OrderedDict[
        str,
        Path,
    ] = OrderedDict()

    for adapter in adapters_10:
        method_id = adapter_label(
            "R10",
            adapter,
        )
        adapter_method_dirs[method_id] = adapter
        reduced_method_names[method_id] = (
            f"10% Random LoRA "
            f"({method_id.split('_', 1)[1]})"
        )

    for adapter in adapters_5:
        method_id = adapter_label(
            "R5",
            adapter,
        )
        adapter_method_dirs[method_id] = adapter
        reduced_method_names[method_id] = (
            f"5% Random LoRA "
            f"({method_id.split('_', 1)[1]})"
        )

    method_names = {
        **FIXED_EXPLANATION_NAMES,
        **reduced_method_names,
    }

    protocol = {
        "experiment": (
            "fitness_home_final_blind_end_to_end_v1"
        ),
        "status": (
            "frozen_before_final_blind_result_inspection"
        ),
        "one_time_final_blind_test": True,
        "no_further_method_tuning_after_completion": True,
        "blind_signature_file": str(
            BLIND_SIGNATURE_FILE
        ),
        "blind_signature_sha256": (
            preflight_data[
                "blind_signature_sha256"
            ]
        ),
        "blind_source_sha_verified": (
            preflight_data[
                "blind_source_sha_verified"
            ]
        ),
        "blind_signature_count": 250,
        "retrieval_queries_per_signature": len(
            QUERY_TEMPLATES
        ),
        "retrieval_query_count": (
            250 * len(QUERY_TEMPLATES)
        ),
        "explanation_sample_count": 250,
        "overlap_checks": preflight_data[
            "overlap"
        ],
        "final_retriever": (
            "B6_HybridRRF_ConstraintRerank"
        ),
        "retrieval_parameters": {
            "bm25_candidate_k": CANDIDATE_K,
            "dense_candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "fused_candidate_k": CANDIDATE_K,
            "constraint_rerank_top_k": TOP_K,
        },
        "explanation_methods": {
            method_id: method_names[method_id]
            for method_id in (
                *FIXED_EXPLANATION_METHODS,
                *adapter_method_dirs.keys(),
            )
        },
        "full_adapter": str(
            FULL_ADAPTER_DIR
        ),
        "reduced_adapters": {
            method_id: str(path)
            for method_id, path in (
                adapter_method_dirs.items()
            )
        },
        "generation": {
            "seed": SEED,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "tinyllama_batch_size": (
                args.tinyllama_batch_size
            ),
            "teacher_batch_size": (
                args.teacher_batch_size
            ),
        },
        "reference_policy": (
            "Deterministic structured reference generated from the "
            "selected restaurant's database values and constraint checks. "
            "ROUGE-L is secondary; structured exact metrics are primary."
        ),
        "blind_test_used": True,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = (
        OUT_ROOT / "final_blind_protocol.json"
    )

    if protocol_path.exists():
        existing = read_json(protocol_path)
        if existing != protocol:
            raise RuntimeError(
                "Existing final blind protocol differs from the "
                "current frozen configuration. Do not overwrite."
            )
    else:
        protocol_path.write_text(
            json.dumps(
                protocol,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("=" * 80)
    print("FITNESS HOME — FINAL BLIND END-TO-END EVALUATION")
    print("=" * 80)
    print("Blind signatures       : 250")
    print(
        "Retrieval blind queries:",
        250 * len(QUERY_TEMPLATES),
    )
    print("Explanation samples    : 250")
    print("Final retriever         : B6 Hybrid RRF")
    print(
        "Explanation methods    :",
        [
            *FIXED_EXPLANATION_METHODS,
            *adapter_method_dirs.keys(),
        ],
    )
    print("Blind/Main overlap      : 0")
    print("Blind/Development overlap: 0")
    print("One-time final test     : YES")
    print("Output                  :", OUT_ROOT)

    if args.preflight_only:
        print()
        print("FINAL BLIND PREFLIGHT PASSED")
        print("No retrieval or model output was generated.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the final blind model evaluation."
        )

    step54 = load_module(
        STEP54_FILE,
        "fh_step54_final_blind",
    )
    step55 = load_module(
        STEP55_FILE,
        "fh_step55_final_blind",
    )
    step52 = load_module(
        STEP52_FILE,
        "fh_step52_final_blind",
    )

    step10_file = next(
        (
            path
            for path in step54.STEP10_CANDIDATES
            if path.exists()
        ),
        None,
    )
    if step10_file is None:
        raise FileNotFoundError(
            "Could not locate the frozen Step-10 query builder."
        )

    step10 = load_module(
        step10_file,
        "fh_step10_final_blind",
    )
    step2 = step10.load_module(
        step10.SOURCE_STEP2,
        "fh_step2_final_blind",
    )

    if str(BACKEND) not in sys.path:
        sys.path.insert(
            0,
            str(BACKEND),
        )

    from rag.retriever import retrieve

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    restaurants = step2.load_restaurants()
    if len(restaurants) != 4996:
        raise RuntimeError(
            f"Expected 4,996 restaurants, found {len(restaurants)}."
        )

    corpus_texts = [
        step54.document_text_of(
            dict(record)
        )
        for record in restaurants
    ]

    print("Building frozen BM25 index...")
    bm25 = step54.BM25Index(
        corpus_texts
    )

    retrieval_cache_path = (
        OUT_ROOT
        / "blind_retrieval_per_query.jsonl"
    )
    retrieval_existing = (
        step54.load_completed(
            retrieval_cache_path
        )
    )

    oracle_cache_path = (
        OUT_ROOT
        / "blind_oracle_by_signature.jsonl"
    )
    oracle_existing = (
        step54.load_completed(
            oracle_cache_path
        )
    )

    blind_query_rows: list[
        dict[str, Any]
    ] = []
    retrieval_method_rows: dict[
        str,
        list[dict[str, Any]],
    ] = {
        method_id: []
        for method_id in (
            RETRIEVAL_METHOD_ORDER
        )
    }

    canonical_b6_result: dict[
        str,
        dict[str, Any],
    ] = {}

    for signature_index, signature_row in enumerate(
        blind_rows,
        start=1,
    ):
        signature_id = signature_row[
            "constraint_signature_id"
        ]
        constraints = signature_row[
            "constraints"
        ]

        if signature_id not in oracle_existing:
            evaluated_all = (
                step10.evaluate_results(
                    step2=step2,
                    raw_results=list(
                        restaurants
                    ),
                    constraints=constraints,
                    relevant_ids=set(),
                )
            )
            relevant_ids = sorted({
                str(
                    step54.restaurant_id_of(
                        result,
                        step10,
                    )
                )
                for result in evaluated_all
                if str(
                    result.get(
                        "match_type",
                        "",
                    )
                ).lower()
                == "full"
            })

            if not relevant_ids:
                raise RuntimeError(
                    f"Blind signature {signature_id} has no full-match "
                    "restaurant. The final blind protocol is invalid."
                )

            structured_top5 = (
                step54.rerank_by_constraints(
                    evaluated_all,
                    constraints,
                    top_k=TOP_K,
                )
            )
            oracle_row = {
                "query_id": signature_id,
                "constraint_signature_id": signature_id,
                "constraints": constraints,
                "ground_truth_restaurant_ids": relevant_ids,
                "structured_top5": structured_top5,
            }
            append_jsonl(
                oracle_cache_path,
                oracle_row,
            )
            oracle_existing[
                signature_id
            ] = oracle_row

        oracle_row = oracle_existing[
            signature_id
        ]
        relevant_ids = list(
            oracle_row[
                "ground_truth_restaurant_ids"
            ]
        )

        for paraphrase_index in range(
            len(QUERY_TEMPLATES)
        ):
            query_id = (
                f"BLINDQ"
                f"{signature_index:04d}"
                f"_P{paraphrase_index + 1:02d}"
            )
            query_text = render_query(
                constraints,
                paraphrase_index,
            )
            query_row = {
                "query_id": query_id,
                "constraint_signature_id": signature_id,
                "query": query_text,
                "constraints": constraints,
                "ground_truth_restaurant_ids": relevant_ids,
            }
            blind_query_rows.append(query_row)

            if query_id not in retrieval_existing:
                document_indices, bm25_scores = (
                    bm25.top_indices(
                        query_text,
                        CANDIDATE_K,
                    )
                )
                raw_bm25 = []
                for rank, (
                    document_index,
                    score,
                ) in enumerate(
                    zip(
                        document_indices,
                        bm25_scores,
                    ),
                    start=1,
                ):
                    result = dict(
                        restaurants[
                            int(document_index)
                        ]
                    )
                    result["rank"] = rank
                    result[
                        "bm25_score"
                    ] = float(score)
                    result[
                        "similarity_score"
                    ] = float(score)
                    raw_bm25.append(result)

                bm25_evaluated = (
                    step54.evaluate_raw_results(
                        raw_bm25,
                        query_row,
                        step10,
                        step2,
                    )
                )

                raw_dense = retrieve(
                    query_text,
                    top_k=CANDIDATE_K,
                )
                dense_evaluated = (
                    step54.evaluate_raw_results(
                        raw_dense,
                        query_row,
                        step10,
                        step2,
                    )
                )

                fused_candidates = (
                    step55.build_rrf_candidates(
                        bm25_evaluated,
                        dense_evaluated,
                        step54,
                        step10,
                    )
                )

                method_results = {
                    "B1_BM25": (
                        bm25_evaluated[
                            :TOP_K
                        ]
                    ),
                    "B2_BM25_ConstraintRerank": (
                        step54.rerank_by_constraints(
                            bm25_evaluated,
                            constraints,
                            top_k=TOP_K,
                        )
                    ),
                    "B3_DenseBGE": (
                        dense_evaluated[
                            :TOP_K
                        ]
                    ),
                    "B4_DenseBGE_ConstraintRerank": (
                        step54.rerank_by_constraints(
                            dense_evaluated,
                            constraints,
                            top_k=TOP_K,
                        )
                    ),
                    "B6_HybridRRF_ConstraintRerank": (
                        step54.rerank_by_constraints(
                            fused_candidates,
                            constraints,
                            top_k=TOP_K,
                        )
                    ),
                    "B5_StructuredConstraintOracle": (
                        oracle_row[
                            "structured_top5"
                        ]
                    ),
                }

                output = {
                    **query_row,
                    "methods": {},
                }
                for method_id, results in (
                    method_results.items()
                ):
                    metrics = (
                        step54.per_query_metrics(
                            query_row,
                            results,
                            step10,
                        )
                    )
                    output["methods"][
                        method_id
                    ] = {
                        "metrics": metrics,
                        "retrieval_results": results,
                    }

                append_jsonl(
                    retrieval_cache_path,
                    output,
                )
                retrieval_existing[
                    query_id
                ] = output

            output = retrieval_existing[
                query_id
            ]

            for method_id in (
                RETRIEVAL_METHOD_ORDER
            ):
                retrieval_method_rows[
                    method_id
                ].append(
                    output["methods"][
                        method_id
                    ]["metrics"]
                )

            if paraphrase_index == 0:
                b6_results = (
                    output["methods"][
                        "B6_HybridRRF_ConstraintRerank"
                    ][
                        "retrieval_results"
                    ]
                )
                if not b6_results:
                    raise RuntimeError(
                        f"B6 returned no result for {query_id}"
                    )
                canonical_b6_result[
                    signature_id
                ] = b6_results[0]

        if (
            signature_index % 10 == 0
            or signature_index
            == len(blind_rows)
        ):
            print(
                f"[final blind retrieval] "
                f"{signature_index}/250 signatures "
                f"({signature_index * len(QUERY_TEMPLATES)}/1000 queries)",
                flush=True,
            )

    query_set_path = (
        OUT_ROOT
        / "blind_query_set_1000.jsonl"
    )
    write_jsonl(
        query_set_path,
        blind_query_rows,
    )

    retrieval_summaries = {
        method_id: (
            step55.aggregate_with_id_diversity(
                rows,
                step54,
            )
        )
        for method_id, rows in (
            retrieval_method_rows.items()
        )
    }

    retrieval_table_path = (
        OUT_ROOT
        / "final_blind_retrieval_table.md"
    )
    write_retrieval_table(
        retrieval_table_path,
        retrieval_summaries,
    )

    blind_explanation_records = []
    retrieval_full_by_sample: dict[
        str,
        bool,
    ] = {}

    for signature_row in blind_rows:
        signature_id = signature_row[
            "constraint_signature_id"
        ]
        canonical_query = render_query(
            signature_row["constraints"],
            0,
        )
        record = build_explanation_record(
            signature_row,
            canonical_query,
            canonical_b6_result[
                signature_id
            ],
            step54,
            step10,
        )
        blind_explanation_records.append(
            record
        )
        retrieval_full_by_sample[
            record["sample_id"]
        ] = (
            str(
                record["metadata"][
                    "match_type"
                ]
            ).lower()
            == "full"
        )

    explanation_benchmark_path = (
        OUT_ROOT
        / "blind_explanation_benchmark_250.jsonl"
    )
    write_jsonl(
        explanation_benchmark_path,
        blind_explanation_records,
    )

    prompts = read_json(PROMPT_FILE)
    common_system_prompt = prompts[
        "common_system_prompt"
    ]
    no_rag_template = prompts[
        "m2_no_rag_user_prompt"
    ]

    all_method_ids = [
        *FIXED_EXPLANATION_METHODS,
        *adapter_method_dirs.keys(),
    ]
    step52.METHOD_NAMES.update(
        method_names
    )

    prediction_paths = {
        method_id: (
            OUT_ROOT
            / f"{method_id.lower()}_predictions.jsonl"
        )
        for method_id in all_method_ids
    }

    ensure_template_predictions(
        blind_explanation_records,
        prediction_paths["M1"],
        method_names["M1"],
    )

    tiny_tokenizer = (
        step52.load_tiny_tokenizer()
    )
    base_model = step52.load_base_model(
        BASE_MODEL
    )

    step52.generate_method_predictions(
        method_id="M2",
        model=base_model,
        tokenizer=tiny_tokenizer,
        records=blind_explanation_records,
        system_prompt=common_system_prompt,
        user_content_builder=(
            lambda record: (
                step52.build_no_rag_user_content(
                    record,
                    no_rag_template,
                )
            )
        ),
        batch_size=args.tinyllama_batch_size,
        output_path=prediction_paths["M2"],
    )
    step52.generate_method_predictions(
        method_id="M3",
        model=base_model,
        tokenizer=tiny_tokenizer,
        records=blind_explanation_records,
        system_prompt=common_system_prompt,
        user_content_builder=(
            step52.build_evidence_user_content
        ),
        batch_size=args.tinyllama_batch_size,
        output_path=prediction_paths["M3"],
    )

    step52.release_model(base_model)
    del tiny_tokenizer
    gc.collect()

    full_tokenizer = (
        step52.load_tiny_tokenizer()
    )
    full_model = (
        step52.load_lora_model()
    )
    step52.generate_method_predictions(
        method_id="M4",
        model=full_model,
        tokenizer=full_tokenizer,
        records=blind_explanation_records,
        system_prompt=common_system_prompt,
        user_content_builder=(
            step52.build_evidence_user_content
        ),
        batch_size=args.tinyllama_batch_size,
        output_path=prediction_paths["M4"],
    )
    step52.release_model(full_model)
    del full_tokenizer
    gc.collect()

    for method_id, adapter_dir in (
        adapter_method_dirs.items()
    ):
        tokenizer = (
            step52.load_tiny_tokenizer()
        )
        model = load_generic_lora_model(
            adapter_dir,
            step52,
        )
        step52.generate_method_predictions(
            method_id=method_id,
            model=model,
            tokenizer=tokenizer,
            records=blind_explanation_records,
            system_prompt=common_system_prompt,
            user_content_builder=(
                step52.build_evidence_user_content
            ),
            batch_size=args.tinyllama_batch_size,
            output_path=prediction_paths[
                method_id
            ],
        )
        step52.release_model(model)
        del tokenizer
        gc.collect()

    teacher_tokenizer = (
        step52.load_teacher_tokenizer()
    )
    teacher_model = (
        step52.load_base_model(
            TEACHER_MODEL
        )
    )
    step52.generate_method_predictions(
        method_id="M5",
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        records=blind_explanation_records,
        system_prompt=common_system_prompt,
        user_content_builder=(
            step52.build_evidence_user_content
        ),
        batch_size=args.teacher_batch_size,
        output_path=prediction_paths["M5"],
    )
    step52.release_model(teacher_model)
    del teacher_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    strict_filter = (
        load_module(
            FILTER_FILE,
            "fh_filter_final_blind",
        )
        if FILTER_FILE.exists()
        else None
    )

    scored_by_method: OrderedDict[
        str,
        list[dict[str, Any]],
    ] = OrderedDict()
    summaries_by_method: OrderedDict[
        str,
        dict[str, Any],
    ] = OrderedDict()

    for method_id in all_method_ids:
        scored_path = (
            OUT_ROOT
            / f"{method_id.lower()}_predictions_scored.jsonl"
        )
        scored = step52.score_method(
            method_id,
            prediction_paths[
                method_id
            ],
            blind_explanation_records,
            strict_filter,
            scored_path,
        )
        scored_by_method[
            method_id
        ] = scored
        summaries_by_method[
            method_id
        ] = aggregate_end_to_end(
            scored,
            retrieval_full_by_sample,
            step52,
        )

    explanation_table_path = (
        OUT_ROOT
        / "final_blind_explanation_table.md"
    )
    write_explanation_table(
        explanation_table_path,
        summaries_by_method,
        method_names,
    )

    ids_10 = [
        method_id
        for method_id in (
            adapter_method_dirs
        )
        if method_id.startswith(
            "R10_"
        )
    ]
    ids_5 = [
        method_id
        for method_id in (
            adapter_method_dirs
        )
        if method_id.startswith(
            "R5_"
        )
    ]

    reduced_groups = {
        "10% Random LoRA": (
            reduced_group_summary(
                ids_10,
                summaries_by_method,
            )
        ),
        "5% Random LoRA": (
            reduced_group_summary(
                ids_5,
                summaries_by_method,
            )
        ),
    }

    reduced_table_path = (
        OUT_ROOT
        / "final_blind_reduced_data_table.md"
    )
    write_reduced_table(
        reduced_table_path,
        reduced_groups,
    )

    # Paired final-blind comparisons.
    method_ids_sorted = list(
        scored_by_method
    )
    significance: dict[
        str,
        Any,
    ] = {}

    def paired_binary(
        first_id: str,
        second_id: str,
        metric_key: str,
    ) -> dict[str, Any]:
        first = {
            row["sample_id"]: row
            for row in scored_by_method[
                first_id
            ]
        }
        second = {
            row["sample_id"]: row
            for row in scored_by_method[
                second_id
            ]
        }
        ids = sorted(first)
        return exact_mcnemar(
            [
                bool(
                    first[sample_id][
                        "metrics"
                    ][metric_key]
                )
                for sample_id in ids
            ],
            [
                bool(
                    second[sample_id][
                        "metrics"
                    ][metric_key]
                )
                for sample_id in ids
            ],
        )

    for first_id, second_id in (
        ("M3", "M4"),
        ("M4", "M5"),
    ):
        significance[
            f"{first_id}_vs_{second_id}"
        ] = {
            "all_constraint_exact": (
                paired_binary(
                    first_id,
                    second_id,
                    "all_constraint_exact",
                )
            ),
            "faithfulness": (
                paired_binary(
                    first_id,
                    second_id,
                    "faithfulness_pass",
                )
            ),
        }

    significance_path = (
        OUT_ROOT
        / "final_blind_pairwise_significance.json"
    )
    significance_path.write_text(
        json.dumps(
            significance,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_summary = {
        "experiment": (
            "fitness_home_final_blind_end_to_end_v1"
        ),
        "status": "final_blind_complete",
        "one_time_final_blind_test": True,
        "no_further_method_tuning": True,
        "blind_test_used": True,
        "blind_signature_count": 250,
        "retrieval_query_count": (
            250 * len(QUERY_TEMPLATES)
        ),
        "explanation_sample_count": 250,
        "retrieval": retrieval_summaries,
        "explanation": summaries_by_method,
        "reduced_data": reduced_groups,
        "significance": significance,
        "final_system": {
            "retriever": (
                "B6_HybridRRF_ConstraintRerank"
            ),
            "explainer": (
                "M4_Final_Hybrid_RAG_LoRA_TinyLlama_100pct"
            ),
            "retrieval_full_at_1": (
                summaries_by_method[
                    "M4"
                ][
                    "retrieval_full_at_1_rate"
                ]
            ),
            "end_to_end_exact_accuracy": (
                summaries_by_method[
                    "M4"
                ][
                    "end_to_end_exact_accuracy"
                ]
            ),
            "end_to_end_full_and_faithful_rate": (
                summaries_by_method[
                    "M4"
                ][
                    "end_to_end_full_and_faithful_rate"
                ]
            ),
        },
    }

    summary_path = (
        OUT_ROOT
        / "final_blind_evaluation_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            final_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_files = [
        protocol_path,
        query_set_path,
        oracle_cache_path,
        retrieval_cache_path,
        retrieval_table_path,
        explanation_benchmark_path,
        explanation_table_path,
        reduced_table_path,
        significance_path,
        summary_path,
        *prediction_paths.values(),
        *[
            OUT_ROOT
            / f"{method_id.lower()}_predictions_scored.jsonl"
            for method_id in all_method_ids
        ],
    ]

    checksum_path = (
        OUT_ROOT / "SHA256SUMS.txt"
    )
    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for path in output_files:
            file.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    LOCK_FILE.write_text(
        json.dumps(
            {
                "status": (
                    "FINAL_BLIND_COMPLETE_LOCKED"
                ),
                "completed_unix_time": time.time(),
                "summary_sha256": sha256_file(
                    summary_path
                ),
                "no_further_tuning": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL BLIND END-TO-END EVALUATION COMPLETE")
    print("=" * 80)

    b6 = retrieval_summaries[
        "B6_HybridRRF_ConstraintRerank"
    ]
    print(
        "B6 retrieval: "
        f"P@5={b6['mean_precision_at_5']:.4f} "
        f"R@5={b6['mean_recall_at_5']:.4f} "
        f"MRR={b6['mrr']:.4f} "
        f"nDCG@5={b6['mean_ndcg_at_5']:.4f} "
        f"Hit@5={b6['hit_at_5_rate']:.2%} "
        f"Full@1={b6['full_match_at_1_rate']:.2%}"
    )

    for method_id in (
        "M1",
        "M3",
        "M4",
        "M5",
    ):
        summary = summaries_by_method[
            method_id
        ]
        print(
            f"{method_id} "
            f"Exact={summary['all_constraint_exact_accuracy']:.2%} "
            f"StateF1={summary['constraint_state_macro_f1']:.2%} "
            f"Numeric={summary['numeric_relation_accuracy']:.2%} "
            f"Faith={summary['faithfulness_rate']:.2%} "
            f"ROUGE-L={summary['mean_rouge_l_f1']:.4f} "
            f"E2EExact={summary['end_to_end_exact_accuracy']:.2%}"
        )

    print(
        "Final system E2E Exact :",
        f"{summaries_by_method['M4']['end_to_end_exact_accuracy']:.2%}",
    )
    print(
        "Final system Full+Faith:",
        f"{summaries_by_method['M4']['end_to_end_full_and_faithful_rate']:.2%}",
    )
    print("Retrieval table       :", retrieval_table_path)
    print("Explanation table     :", explanation_table_path)
    print("Reduced-data table    :", reduced_table_path)
    print("Summary               :", summary_path)
    print("Lock                  :", LOCK_FILE)
    print("Blind test used       : YES")
    print("Further tuning        : FORBIDDEN")


if __name__ == "__main__":
    main()
