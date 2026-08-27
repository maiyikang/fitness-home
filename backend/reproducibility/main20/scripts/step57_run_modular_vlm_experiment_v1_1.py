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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import set_seed

HERE = Path(__file__).resolve().parent

STEP56_FILE = HERE / "step56_run_vlm_multiconstraint_experiment.py"
STEP52_FILE = HERE / "step52_run_explanation_baseline_m1_m5.py"

BENCHMARK_DIR = HERE / "45_vlm_multiconstraint_benchmark"
BENCHMARK_FILE = BENCHMARK_DIR / "vlm_benchmark_300.jsonl"
BENCHMARK_SHA_FILE = BENCHMARK_DIR / "SHA256SUMS_BENCHMARK.txt"

VLM_V1_DIR = HERE / "46_vlm_multiconstraint_eval" / "development_300"
VLM_V1_SHA_FILE = VLM_V1_DIR / "SHA256SUMS.txt"

OUT_ROOT = HERE / "47_modular_vlm_eval"

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
SEED = 20260819
MAX_NEW_TOKENS = 120

METHOD_ORDER = (
    "T0",
    "V3",
    "P0",
    "P1",
    "MV0",
    "MV1",
)

METHOD_NAMES = {
    "T0": "Text-only RAG+LoRA TinyLlama",
    "V3": "Monolithic VLM + RAG + conflict prompt (Step 56)",
    "P0": "Existing general image-only VLM parser (Step 56 V1)",
    "P1": "Focused image-only visual parser",
    "MV0": "Focused parser + RAG+LoRA without visual gate",
    "MV1": "Focused parser + deterministic visual gate + RAG+LoRA",
}

CUISINES = (
    "American",
    "Chinese",
    "Greek",
    "Indian",
    "Italian",
    "Japanese",
    "Korean",
    "Mediterranean",
    "Mexican",
    "Seafood",
    "Thai",
    "Vietnamese",
)

FOCUSED_SYSTEM_PROMPT = (
    "You are a visual food parser. Inspect only the image. "
    "Do not use restaurant names, user requests, nutrition values, or text "
    "outside the image to infer the answer. Return valid JSON only with "
    "exactly four fields: visual_cuisine_guess, visual_quality, confidence, "
    "visual_description. visual_cuisine_guess must be exactly one of: "
    + ", ".join(CUISINES)
    + ", or uncertain. visual_quality must be clear, noisy, or uncertain. "
    "confidence must be a number from 0 to 1. visual_description must be one "
    "short factual sentence about visible food appearance only. Never infer "
    "calories, protein, fibre, health effects, restaurant identity, or fitness "
    "benefits."
)

FOCUSED_USER_PROMPT = (
    "Inspect this food image only. Identify the most likely cuisine category "
    "from the allowed list, assess whether the image is clear or noisy, and "
    "give one short visual description. Return JSON only."
)

JSON_FIELDS = (
    "visual_cuisine_guess",
    "visual_quality",
    "confidence",
    "visual_description",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen two-stage Modular VLM experiment: focused visual "
            "parsing, deterministic conflict/noise gating, and the existing "
            "RAG+LoRA explanation model."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test limit. Full run uses all 300 frozen VLM samples.",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
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


def parse_sha256_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(maxsplit=1)
            values[name.strip()] = digest.strip()
    return values


def verify_frozen_benchmark() -> None:
    if not BENCHMARK_SHA_FILE.exists():
        raise FileNotFoundError(BENCHMARK_SHA_FILE)

    expected = parse_sha256_file(BENCHMARK_SHA_FILE)
    observed: dict[str, str] = {
        BENCHMARK_FILE.name: sha256_file(BENCHMARK_FILE),
        "vlm_benchmark_protocol.json": sha256_file(
            BENCHMARK_DIR / "vlm_benchmark_protocol.json"
        ),
    }
    for image_path in sorted((BENCHMARK_DIR / "images").glob("*.jpg")):
        observed[str(image_path.relative_to(BENCHMARK_DIR))] = sha256_file(
            image_path
        )

    if expected != observed:
        mismatch = sorted(
            name
            for name in expected.keys() & observed.keys()
            if expected[name] != observed[name]
        )
        raise RuntimeError(
            "Frozen VLM benchmark checksum failure: "
            f"missing={sorted(set(expected)-set(observed))[:5]} "
            f"extra={sorted(set(observed)-set(expected))[:5]} "
            f"mismatch={mismatch[:5]}"
        )


def verify_step56_outputs() -> None:
    if not VLM_V1_SHA_FILE.exists():
        raise FileNotFoundError(VLM_V1_SHA_FILE)

    expected = parse_sha256_file(VLM_V1_SHA_FILE)
    required = (
        "t0_predictions_scored.jsonl",
        "v1_predictions.jsonl",
        "v3_predictions_scored.jsonl",
        "vlm_evaluation_summary.json",
    )
    for filename in required:
        path = VLM_V1_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)
        if filename not in expected:
            raise RuntimeError(
                f"{filename} is absent from frozen Step 56 SHA256SUMS."
            )
        if sha256_file(path) != expected[filename]:
            raise RuntimeError(
                f"Step 56 checksum mismatch for {filename}."
            )


def load_completed(
    path: Path,
    allowed_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        benchmark_id = str(row.get("benchmark_id", ""))
        if benchmark_id not in allowed_ids:
            raise RuntimeError(
                f"Unexpected benchmark_id in {path}: {benchmark_id}"
            )
        if benchmark_id in result:
            raise RuntimeError(
                f"Duplicate benchmark_id in {path}: {benchmark_id}"
            )
        result[benchmark_id] = row
    return result


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


def parse_focused_json(raw: str) -> dict[str, Any]:
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
            "confidence": 0.0,
            "visual_description": "",
        }

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "json_parse_success": all(field in parsed for field in JSON_FIELDS),
        "visual_cuisine_guess": str(
            parsed.get("visual_cuisine_guess", "uncertain")
        ).strip(),
        "visual_quality": str(
            parsed.get("visual_quality", "uncertain")
        ).strip().lower(),
        "confidence": min(1.0, max(0.0, confidence)),
        "visual_description": str(
            parsed.get("visual_description", "")
        ).strip(),
    }


def focused_chat_text(processor: Any) -> str:
    messages = [
        {
            "role": "system",
            "content": FOCUSED_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": FOCUSED_USER_PROMPT},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_focused_parser(
    benchmark_rows: Sequence[dict[str, Any]],
    processor: Any,
    model: Any,
    output_path: Path,
    step56: Any,
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
        print("[P1] Focused visual parser already complete; skipping.")
        return

    prompt = focused_chat_text(processor)
    completed = len(existing)
    print(
        f"[P1] Existing={len(existing)} Pending={len(pending)}"
    )

    for benchmark_row in pending:
        benchmark_id = str(benchmark_row["benchmark_id"])
        image_path = BENCHMARK_DIR / str(benchmark_row["image_file"])
        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = step56.move_inputs(dict(inputs), model)
        input_width = int(inputs["input_ids"].shape[-1])

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

        raw_output = processor.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_focused_json(raw_output)

        append_jsonl(
            output_path,
            {
                "benchmark_id": benchmark_id,
                "source_sample_id": benchmark_row["source_sample_id"],
                "method_id": "P1",
                "method_name": METHOD_NAMES["P1"],
                "raw_output": raw_output,
                "parsed": parsed,
                "generation_seconds": elapsed,
            },
        )

        completed += 1
        print(
            f"[P1] {completed:03d}/{len(benchmark_rows):03d} "
            f"{benchmark_id}",
            flush=True,
        )
        image.close()


def parser_fields(
    parsed: dict[str, Any],
    step56: Any,
) -> dict[str, Any]:
    cuisine = step56.canonical_cuisine(
        parsed.get("visual_cuisine_guess")
    )
    quality = step56.normalise_quality(
        parsed.get("visual_quality")
    )
    return {
        "predicted_cuisine": cuisine,
        "predicted_quality": quality,
    }


def deterministic_relation(
    predicted_cuisine: str | None,
    target_cuisine: str,
) -> str:
    if predicted_cuisine is None:
        return "uncertain"
    if predicted_cuisine == target_cuisine:
        return "aligned"
    return "conflict"


def parser_metrics(
    parsed: dict[str, Any],
    benchmark_row: dict[str, Any],
    step56: Any,
) -> dict[str, Any]:
    fields = parser_fields(parsed, step56)
    predicted_cuisine = fields["predicted_cuisine"]
    predicted_quality = fields["predicted_quality"]
    target_cuisine = str(benchmark_row["target_cuisine"])
    image_cuisine = str(benchmark_row["image_cuisine"])
    scenario = str(benchmark_row["scenario"])

    relation = deterministic_relation(
        predicted_cuisine,
        target_cuisine,
    )
    cuisine_correct = predicted_cuisine == image_cuisine

    if scenario == "aligned_noisy":
        relation_safe = relation in ("aligned", "uncertain")
        quality_safe = predicted_quality in ("noisy", "uncertain")
    elif scenario == "aligned_clear":
        relation_safe = relation == "aligned"
        quality_safe = predicted_quality == "clear"
    elif scenario == "conflict_clear":
        relation_safe = relation == "conflict"
        quality_safe = predicted_quality == "clear"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return {
        **fields,
        "deterministic_relation": relation,
        "visual_cuisine_accuracy": cuisine_correct,
        "visual_relation_safety_accuracy": relation_safe,
        "visual_quality_safety_accuracy": quality_safe,
        "json_parse_success": bool(
            parsed.get("json_parse_success", False)
        ),
    }


def policy_decision(
    metrics: dict[str, Any],
    gated: bool,
) -> dict[str, Any]:
    relation = metrics["deterministic_relation"]
    quality = metrics["predicted_quality"]

    if not gated:
        return {
            "policy_action": "use_visual_without_gate",
            "database_priority": False,
            "visual_suppressed": False,
        }

    if quality in ("noisy", "uncertain"):
        return {
            "policy_action": "suppress_noisy_or_uncertain",
            "database_priority": True,
            "visual_suppressed": True,
        }
    if relation == "conflict":
        return {
            "policy_action": "suppress_conflict",
            "database_priority": True,
            "visual_suppressed": True,
        }
    if relation == "aligned":
        return {
            "policy_action": "use_as_complementary",
            "database_priority": True,
            "visual_suppressed": False,
        }
    return {
        "policy_action": "suppress_uncertain",
        "database_priority": True,
        "visual_suppressed": True,
    }


def expected_policy_safe(
    benchmark_row: dict[str, Any],
    parser_result: dict[str, Any],
    policy: dict[str, Any],
) -> bool:
    scenario = str(benchmark_row["scenario"])
    cuisine_correct = bool(
        parser_result["visual_cuisine_accuracy"]
    )
    relation = parser_result["deterministic_relation"]
    quality = parser_result["predicted_quality"]
    action = policy["policy_action"]

    if scenario == "aligned_clear":
        return bool(
            cuisine_correct
            and relation == "aligned"
            and quality == "clear"
            and action == "use_as_complementary"
        )
    if scenario == "aligned_noisy":
        return bool(
            quality in ("noisy", "uncertain")
            and action in (
                "suppress_noisy_or_uncertain",
                "suppress_uncertain",
            )
        )
    if scenario == "conflict_clear":
        return bool(
            cuisine_correct
            and relation == "conflict"
            and quality == "clear"
            and action == "suppress_conflict"
        )
    return False


def visual_addendum(
    benchmark_row: dict[str, Any],
    parser_result: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    target = str(benchmark_row["target_cuisine"])
    predicted = parser_result["predicted_cuisine"]
    action = policy["policy_action"]

    if action == "use_as_complementary":
        cuisine_text = predicted or "the requested"
        return (
            f"The image appears visually consistent with {cuisine_text} "
            "cuisine and is used only as complementary visual evidence."
        )
    if action == "suppress_conflict":
        cuisine_text = predicted or "a different"
        return (
            f"The image appears visually inconsistent with the requested "
            f"{target} cuisine and instead resembles {cuisine_text} cuisine; "
            "the image is therefore treated as non-authoritative, while the "
            "recommendation remains grounded in the structured restaurant "
            "evidence."
        )
    if action in ("suppress_noisy_or_uncertain", "suppress_uncertain"):
        return (
            "The image is too noisy or uncertain for reliable visual "
            "confirmation, so the recommendation remains grounded in the "
            "structured restaurant evidence."
        )
    cuisine_text = predicted or "an uncertain cuisine"
    return (
        f"The image appears to show {cuisine_text} and is included as visual "
        "evidence without a conflict or quality gate."
    )


def build_modular_rows(
    method_id: str,
    benchmark_rows: Sequence[dict[str, Any]],
    focused_by_id: dict[str, dict[str, Any]],
    t0_by_id: dict[str, dict[str, Any]],
    step56: Any,
    gated: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for benchmark_row in benchmark_rows:
        benchmark_id = str(benchmark_row["benchmark_id"])
        parser_prediction = focused_by_id[benchmark_id]
        parsed = parser_prediction["parsed"]
        p_metrics = parser_metrics(
            parsed,
            benchmark_row,
            step56,
        )
        policy = policy_decision(
            p_metrics,
            gated=gated,
        )
        gate_safe = expected_policy_safe(
            benchmark_row,
            p_metrics,
            policy,
        )
        base_row = t0_by_id[benchmark_id]
        base_metrics = base_row["explanation_metrics"]
        addendum = visual_addendum(
            benchmark_row,
            p_metrics,
            policy,
        )
        final_explanation = (
            f"{addendum} {base_row['parsed']['explanation']}"
        ).strip()

        scenario = str(benchmark_row["scenario"])
        if scenario == "conflict_clear":
            conflict_exact = bool(
                p_metrics["visual_cuisine_accuracy"]
                and p_metrics[
                    "visual_relation_safety_accuracy"
                ]
                and p_metrics[
                    "visual_quality_safety_accuracy"
                ]
                and gate_safe
                and policy["database_priority"]
                and base_metrics["faithfulness_pass"]
            )
        else:
            conflict_exact = None

        if scenario == "aligned_clear":
            visual_component_exact = bool(
                p_metrics["visual_cuisine_accuracy"]
                and p_metrics[
                    "visual_relation_safety_accuracy"
                ]
                and p_metrics[
                    "visual_quality_safety_accuracy"
                ]
                and gate_safe
            )
        elif scenario == "aligned_noisy":
            visual_component_exact = bool(
                p_metrics[
                    "visual_relation_safety_accuracy"
                ]
                and p_metrics[
                    "visual_quality_safety_accuracy"
                ]
                and gate_safe
            )
        else:
            visual_component_exact = bool(
                conflict_exact
            )

        multimodal_exact = bool(
            base_metrics["all_constraint_exact"]
            and visual_component_exact
        )

        database_override_error = bool(
            scenario in ("aligned_noisy", "conflict_clear")
            and not policy["database_priority"]
        )

        rows.append({
            "benchmark_id": benchmark_id,
            "source_sample_id": benchmark_row["source_sample_id"],
            "method_id": method_id,
            "method_name": METHOD_NAMES[method_id],
            "scenario": scenario,
            "target_cuisine": benchmark_row["target_cuisine"],
            "image_cuisine": benchmark_row["image_cuisine"],
            "parser_output": parsed,
            "parser_metrics": p_metrics,
            "policy": policy,
            "gate_safety_accuracy": gate_safe,
            "conflict_resolution_exact": conflict_exact,
            "visual_component_exact": visual_component_exact,
            "multimodal_exact": multimodal_exact,
            "database_override_error": database_override_error,
            "text_explanation_metrics": base_metrics,
            "visual_addendum": addendum,
            "final_multimodal_explanation": final_explanation,
        })

    return rows


def aggregate_boolean(
    rows: Sequence[dict[str, Any]],
    getter: Any,
) -> float | None:
    values = [
        getter(row)
        for row in rows
        if getter(row) is not None
    ]
    if not values:
        return None
    return sum(bool(value) for value in values) / len(values)


def aggregate_parser_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "visual_cuisine_accuracy": aggregate_boolean(
            rows,
            lambda row: row["parser_metrics"][
                "visual_cuisine_accuracy"
            ],
        ),
        "visual_relation_safety_accuracy": aggregate_boolean(
            rows,
            lambda row: row["parser_metrics"][
                "visual_relation_safety_accuracy"
            ],
        ),
        "visual_quality_safety_accuracy": aggregate_boolean(
            rows,
            lambda row: row["parser_metrics"][
                "visual_quality_safety_accuracy"
            ],
        ),
        "json_parse_success_rate": aggregate_boolean(
            rows,
            lambda row: row["parser_metrics"][
                "json_parse_success"
            ],
        ),
    }


def aggregate_modular_rows(
    rows: Sequence[dict[str, Any]],
    step52: Any,
) -> dict[str, Any]:
    text_rows = [
        {
            "sample_id": row["benchmark_id"],
            "generation_seconds_estimate": 0.0,
            "metrics": row["text_explanation_metrics"],
        }
        for row in rows
    ]
    text_summary = dict(step52.aggregate_summary(text_rows))
    text_summary.update({
        "visual_cuisine_accuracy": aggregate_boolean(
            rows,
            lambda row: row["parser_metrics"][
                "visual_cuisine_accuracy"
            ],
        ),
        "visual_relation_safety_accuracy": aggregate_boolean(
            rows,
            lambda row: row["parser_metrics"][
                "visual_relation_safety_accuracy"
            ],
        ),
        "visual_quality_safety_accuracy": aggregate_boolean(
            rows,
            lambda row: row["parser_metrics"][
                "visual_quality_safety_accuracy"
            ],
        ),
        "gate_safety_accuracy": aggregate_boolean(
            rows,
            lambda row: row["gate_safety_accuracy"],
        ),
        "conflict_resolution_exact_accuracy": aggregate_boolean(
            rows,
            lambda row: row["conflict_resolution_exact"],
        ),
        "database_override_error_rate": aggregate_boolean(
            rows,
            lambda row: row["database_override_error"],
        ),
        "multimodal_exact_accuracy": aggregate_boolean(
            rows,
            lambda row: row["multimodal_exact"],
        ),
    })
    return text_summary


def parser_rows_from_predictions(
    predictions: Sequence[dict[str, Any]],
    benchmark_by_id: dict[str, dict[str, Any]],
    step56: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        benchmark_id = str(prediction["benchmark_id"])
        benchmark_row = benchmark_by_id[benchmark_id]
        parsed = prediction["parsed"]
        rows.append({
            "benchmark_id": benchmark_id,
            "scenario": benchmark_row["scenario"],
            "target_cuisine": benchmark_row["target_cuisine"],
            "image_cuisine": benchmark_row["image_cuisine"],
            "parsed": parsed,
            "parser_metrics": parser_metrics(
                parsed,
                benchmark_row,
                step56,
            ),
        })
    return rows


def exact_mcnemar(
    first: Sequence[bool],
    second: Sequence[bool],
) -> dict[str, Any]:
    first_only = sum(
        bool(a) and not bool(b)
        for a, b in zip(first, second)
    )
    second_only = sum(
        bool(b) and not bool(a)
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


def paired_metric(
    first_rows: Sequence[dict[str, Any]],
    second_rows: Sequence[dict[str, Any]],
    getter: Any,
) -> dict[str, Any]:
    first_by_id = {
        row["benchmark_id"]: row
        for row in first_rows
    }
    second_by_id = {
        row["benchmark_id"]: row
        for row in second_rows
    }
    ids = sorted(first_by_id)
    if ids != sorted(second_by_id):
        raise RuntimeError("Paired benchmark IDs do not match.")
    return exact_mcnemar(
        [getter(first_by_id[bid]) for bid in ids],
        [getter(second_by_id[bid]) for bid in ids],
    )


def write_main_table(
    path_csv: Path,
    path_md: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    headers = [
        "Method",
        "Visual Cuisine",
        "Relation Safety",
        "Quality Safety",
        "Gate Safety",
        "Conflict Exact",
        "DB Override Error",
        "Text Exact",
        "State Macro-F1",
        "Numeric Relation",
        "Faithfulness",
        "ROUGE-L",
        "Multimodal Exact",
    ]

    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        if method_id not in summaries:
            continue
        summary = summaries[method_id]
        rows.append({
            "Method": f"{method_id} {METHOD_NAMES[method_id]}",
            "Visual Cuisine": summary.get(
                "visual_cuisine_accuracy"
            ),
            "Relation Safety": summary.get(
                "visual_relation_safety_accuracy"
            ),
            "Quality Safety": summary.get(
                "visual_quality_safety_accuracy"
            ),
            "Gate Safety": summary.get(
                "gate_safety_accuracy"
            ),
            "Conflict Exact": summary.get(
                "conflict_resolution_exact_accuracy"
            ),
            "DB Override Error": summary.get(
                "database_override_error_rate"
            ),
            "Text Exact": summary.get(
                "all_constraint_exact_accuracy"
            ),
            "State Macro-F1": summary.get(
                "constraint_state_macro_f1"
            ),
            "Numeric Relation": summary.get(
                "numeric_relation_accuracy"
            ),
            "Faithfulness": summary.get(
                "faithfulness_rate"
            ),
            "ROUGE-L": summary.get(
                "mean_rouge_l_f1"
            ),
            "Multimodal Exact": summary.get(
                "multimodal_exact_accuracy"
            ),
        })

    with path_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=headers)
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
            else:
                values.append(f"{float(value):.4f}")
        markdown.append("| " + " | ".join(values) + " |")

    path_md.write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    for path in (
        STEP56_FILE,
        STEP52_FILE,
        BENCHMARK_FILE,
        M4_SCORED_FILE := VLM_V1_DIR / "t0_predictions_scored.jsonl",
        VLM_V1_DIR / "v1_predictions.jsonl",
        VLM_V1_DIR / "v3_predictions_scored.jsonl",
        VLM_V1_DIR / "vlm_evaluation_summary.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    verify_frozen_benchmark()
    verify_step56_outputs()

    step56 = load_module(
        STEP56_FILE,
        "fh_step56_modular_vlm",
    )
    step52 = load_module(
        STEP52_FILE,
        "fh_step52_modular_vlm",
    )

    benchmark_all = read_jsonl(BENCHMARK_FILE)
    benchmark_rows = (
        benchmark_all[: args.limit]
        if args.limit is not None
        else benchmark_all
    )
    benchmark_by_id = {
        str(row["benchmark_id"]): row
        for row in benchmark_rows
    }
    allowed_ids = set(benchmark_by_id)

    run_name = (
        f"smoke_{len(benchmark_rows)}"
        if args.limit is not None
        else "development_300"
    )
    run_dir = OUT_ROOT / run_name

    if args.overwrite_run and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment": "fitness_home_modular_vlm_v1",
        "status": "frozen_before_result_inspection",
        "benchmark_samples": len(benchmark_rows),
        "benchmark_sha256": sha256_file(BENCHMARK_FILE),
        "model": MODEL_ID,
        "focused_parser_prompt": FOCUSED_SYSTEM_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "methods": {
            method_id: METHOD_NAMES[method_id]
            for method_id in METHOD_ORDER
        },
        "architecture": (
            "Focused image-only VLM parser -> deterministic visual "
            "conflict/noise gate -> existing RAG+LoRA explanation."
        ),
        "important_note": (
            "The RAG+LoRA text explanation is preserved unchanged. "
            "Visual claims are evaluated against Food-101 image labels, "
            "while restaurant and nutrition claims remain evaluated against "
            "the frozen structured Development evidence."
        ),
        "development_only": True,
        "blind_test_used": False,
    }
    protocol_path = run_dir / "modular_vlm_protocol.json"
    if protocol_path.exists():
        existing = read_json(protocol_path)
        if existing != protocol:
            raise RuntimeError(
                "Existing modular VLM protocol differs. "
                "Use --overwrite-run to restart."
            )
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    print("=" * 78)
    print("FITNESS HOME — TWO-STAGE MODULAR VLM EXPERIMENT")
    print("=" * 78)
    print("Benchmark samples :", len(benchmark_rows))
    print("Model             :", MODEL_ID)
    print("Architecture      : Focused parser -> gate -> RAG+LoRA")
    print("Development only  : YES")
    print("Blind test used   : NO")
    print("Output            :", run_dir)

    focused_path = run_dir / "p1_focused_parser_predictions.jsonl"
    focused_existing = load_completed(focused_path, allowed_ids)

    if len(focused_existing) != len(benchmark_rows):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for focused VLM parser inference."
            )
        processor, model = step56.load_vlm()
        generate_focused_parser(
            benchmark_rows,
            processor,
            model,
            focused_path,
            step56,
        )
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()

    focused_predictions = [
        load_completed(focused_path, allowed_ids)[bid]
        for bid in sorted(allowed_ids)
    ]

    t0_all = read_jsonl(
        VLM_V1_DIR / "t0_predictions_scored.jsonl"
    )
    v1_all = read_jsonl(
        VLM_V1_DIR / "v1_predictions.jsonl"
    )
    v3_all = read_jsonl(
        VLM_V1_DIR / "v3_predictions_scored.jsonl"
    )
    t0_by_id = {
        str(row["benchmark_id"]): row
        for row in t0_all
        if str(row["benchmark_id"]) in allowed_ids
    }
    v1_predictions = [
        row
        for row in v1_all
        if str(row["benchmark_id"]) in allowed_ids
    ]
    v3_rows = [
        row
        for row in v3_all
        if str(row["benchmark_id"]) in allowed_ids
    ]

    if not (
        len(t0_by_id)
        == len(v1_predictions)
        == len(v3_rows)
        == len(benchmark_rows)
    ):
        raise RuntimeError(
            "Existing Step 56 prediction coverage does not match benchmark."
        )

    p0_rows = parser_rows_from_predictions(
        v1_predictions,
        benchmark_by_id,
        step56,
    )
    p1_rows = parser_rows_from_predictions(
        focused_predictions,
        benchmark_by_id,
        step56,
    )
    focused_by_id = {
        str(row["benchmark_id"]): row
        for row in focused_predictions
    }

    m0_rows = build_modular_rows(
        "MV0",
        benchmark_rows,
        focused_by_id,
        t0_by_id,
        step56,
        gated=False,
    )
    m1_rows = build_modular_rows(
        "MV1",
        benchmark_rows,
        focused_by_id,
        t0_by_id,
        step56,
        gated=True,
    )

    write_jsonl(run_dir / "p0_general_parser_scored.jsonl", p0_rows)
    write_jsonl(run_dir / "p1_focused_parser_scored.jsonl", p1_rows)
    write_jsonl(run_dir / "m0_modular_no_gate_scored.jsonl", m0_rows)
    write_jsonl(run_dir / "m1_modular_gate_scored.jsonl", m1_rows)

    step56_summary = read_json(
        VLM_V1_DIR / "vlm_evaluation_summary.json"
    )
    t0_summary = dict(step56_summary["methods"]["T0"])
    v3_summary = dict(step56_summary["methods"]["V3"])

    summaries = {
        "T0": t0_summary,
        "V3": v3_summary,
        "P0": aggregate_parser_rows(p0_rows),
        "P1": aggregate_parser_rows(p1_rows),
        "MV0": aggregate_modular_rows(m0_rows, step52),
        "MV1": aggregate_modular_rows(m1_rows, step52),
    }

    # Give monolithic V3 a consistent database override error field.
    summaries["V3"]["database_override_error_rate"] = (
        1.0 - float(
            summaries["V3"].get(
                "database_priority_accuracy",
                0.0,
            )
        )
    )
    summaries["V3"]["multimodal_exact_accuracy"] = (
        summaries["V3"].get(
            "conflict_resolution_exact_accuracy"
        )
    )

    main_csv = run_dir / "modular_vlm_main_table.csv"
    main_md = run_dir / "modular_vlm_main_table.md"
    write_main_table(main_csv, main_md, summaries)

    # Scenario subgroup tables for the proposed method.
    subgroup_rows: list[dict[str, Any]] = []
    for method_id, method_rows in (
        ("P0", p0_rows),
        ("P1", p1_rows),
        ("MV0", m0_rows),
        ("MV1", m1_rows),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            grouped[str(row["scenario"])].append(row)
        for scenario, rows in sorted(grouped.items()):
            summary = (
                aggregate_parser_rows(rows)
                if method_id in ("P0", "P1")
                else aggregate_modular_rows(rows, step52)
            )
            subgroup_rows.append({
                "method_id": method_id,
                "method_name": METHOD_NAMES[method_id],
                "scenario": scenario,
                **summary,
            })

    subgroup_csv = run_dir / "modular_vlm_subgroup_table.csv"
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
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(subgroup_rows)

    significance = {
        "P0_vs_P1": {
            "direction": "P1 minus P0",
            "visual_cuisine_accuracy": paired_metric(
                p0_rows,
                p1_rows,
                lambda row: row["parser_metrics"][
                    "visual_cuisine_accuracy"
                ],
            ),
            "visual_quality_safety": paired_metric(
                p0_rows,
                p1_rows,
                lambda row: row["parser_metrics"][
                    "visual_quality_safety_accuracy"
                ],
            ),
            "visual_relation_safety": paired_metric(
                p0_rows,
                p1_rows,
                lambda row: row["parser_metrics"][
                    "visual_relation_safety_accuracy"
                ],
            ),
        },
        "MV0_vs_MV1": {
            "direction": "MV1 minus MV0",
            "gate_safety": paired_metric(
                m0_rows,
                m1_rows,
                lambda row: row["gate_safety_accuracy"],
            ),
            "multimodal_exact": paired_metric(
                m0_rows,
                m1_rows,
                lambda row: row["multimodal_exact"],
            ),
        },
    }
    significance_path = run_dir / "modular_vlm_significance.json"
    significance_path.write_text(
        json.dumps(significance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = run_dir / "modular_vlm_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "fitness_home_modular_vlm_v1",
                "development_only": True,
                "blind_test_used": False,
                "benchmark_samples": len(benchmark_rows),
                "methods": summaries,
                "subgroups": subgroup_rows,
                "significance": significance,
                "architecture_conclusion": (
                    "The VLM is restricted to visual parsing. "
                    "A deterministic gate handles conflict/noise, while the "
                    "existing RAG+LoRA model retains responsibility for the "
                    "final evidence-grounded recommendation explanation."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_files = (
        protocol_path,
        focused_path,
        run_dir / "p0_general_parser_scored.jsonl",
        run_dir / "p1_focused_parser_scored.jsonl",
        run_dir / "m0_modular_no_gate_scored.jsonl",
        run_dir / "m1_modular_gate_scored.jsonl",
        main_csv,
        main_md,
        subgroup_csv,
        significance_path,
        summary_path,
    )
    checksum_path = run_dir / "SHA256SUMS.txt"
    with checksum_path.open("w", encoding="utf-8") as file:
        for path in output_files:
            file.write(f"{sha256_file(path)}  {path.name}\n")

    print()
    print("=" * 78)
    print("MODULAR VLM EXPERIMENT COMPLETE")
    print("=" * 78)

    for method_id in METHOD_ORDER:
        summary = summaries[method_id]
        def fmt(key: str) -> str:
            value = summary.get(key)
            return "N/A" if value is None else f"{float(value):.2%}"

        print(
            f"{method_id} "
            f"VisualCuisine={fmt('visual_cuisine_accuracy')} "
            f"Relation={fmt('visual_relation_safety_accuracy')} "
            f"Quality={fmt('visual_quality_safety_accuracy')} "
            f"Gate={fmt('gate_safety_accuracy')} "
            f"Conflict={fmt('conflict_resolution_exact_accuracy')} "
            f"TextExact={fmt('all_constraint_exact_accuracy')} "
            f"Faith={fmt('faithfulness_rate')} "
            f"MultimodalExact={fmt('multimodal_exact_accuracy')}"
        )

    print("Main table       :", main_md)
    print("Subgroup table   :", subgroup_csv)
    print("Significance     :", significance_path)
    print("Summary          :", summary_path)
    print("Blind test used  : NO")


if __name__ == "__main__":
    main()
