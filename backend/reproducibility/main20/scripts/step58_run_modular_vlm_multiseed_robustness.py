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
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import set_seed

HERE = Path(__file__).resolve().parent

STEP56_FILE = HERE / "step56_run_vlm_multiconstraint_experiment.py"
STEP57_FILE = HERE / "step57_run_modular_vlm_experiment_v1_1.py"

ORIGINAL_BENCHMARK_DIR = HERE / "45_vlm_multiconstraint_benchmark"
ORIGINAL_BENCHMARK_FILE = (
    ORIGINAL_BENCHMARK_DIR / "vlm_benchmark_300.jsonl"
)
ORIGINAL_BENCHMARK_SHA = (
    ORIGINAL_BENCHMARK_DIR / "SHA256SUMS_BENCHMARK.txt"
)

STEP56_FROZEN_DIR = (
    HERE
    / "46_vlm_multiconstraint_eval"
    / "development_300_frozen_v1"
)
STEP57_DIR = HERE / "47_modular_vlm_eval" / "development_300"

T0_SCORED_FILE = STEP56_FROZEN_DIR / "t0_predictions_scored.jsonl"
STEP57_SUMMARY_FILE = STEP57_DIR / "modular_vlm_evaluation_summary.json"

OUT_ROOT = HERE / "48_modular_vlm_robustness_multiseed"

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
BASE_SEED = 20260820
IMAGE_SEEDS = tuple(BASE_SEED + offset for offset in range(8))
IMAGES_PER_CUISINE_PER_SEED = 25
MAX_NEW_TOKENS = 120
BOOTSTRAP_REPETITIONS = 5000

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

FOOD101_ROOT = Path(
    os.environ.get(
        "FITNESS_HOME_FOOD101_ROOT",
        str(HERE.parent / "data" / "food-101"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final 8-seed, disjoint-image robustness validation "
            "for the focused visual parser and modular VLM gate."
        )
    )
    parser.add_argument(
        "--overwrite-benchmarks",
        action="store_true",
        help="Rebuild all eight image-seed benchmarks.",
    )
    parser.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="Delete all focused-parser predictions before inference.",
    )
    parser.add_argument(
        "--limit-seeds",
        type=int,
        default=None,
        help="Smoke test using the first N image seeds.",
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


def verify_original_benchmark() -> None:
    if not ORIGINAL_BENCHMARK_SHA.exists():
        raise FileNotFoundError(ORIGINAL_BENCHMARK_SHA)

    expected = parse_sha256_file(ORIGINAL_BENCHMARK_SHA)
    observed: dict[str, str] = {
        ORIGINAL_BENCHMARK_FILE.name: sha256_file(
            ORIGINAL_BENCHMARK_FILE
        ),
        "vlm_benchmark_protocol.json": sha256_file(
            ORIGINAL_BENCHMARK_DIR / "vlm_benchmark_protocol.json"
        ),
    }
    for image_path in sorted(
        (ORIGINAL_BENCHMARK_DIR / "images").glob("*.jpg")
    ):
        observed[str(image_path.relative_to(ORIGINAL_BENCHMARK_DIR))] = (
            sha256_file(image_path)
        )

    if expected != observed:
        raise RuntimeError(
            "Original frozen VLM benchmark failed SHA256 verification."
        )


def seed_dir(seed: int) -> Path:
    return OUT_ROOT / f"image_seed_{seed}"


def seed_benchmark_file(seed: int) -> Path:
    return seed_dir(seed) / "benchmark_300.jsonl"


def seed_sha_file(seed: int) -> Path:
    return seed_dir(seed) / "SHA256SUMS_BENCHMARK.txt"


def benchmark_checksums(seed: int) -> dict[str, str]:
    directory = seed_dir(seed)
    benchmark_file = seed_benchmark_file(seed)

    values = {
        benchmark_file.name: sha256_file(benchmark_file),
    }
    for image_path in sorted((directory / "images").glob("*.jpg")):
        values[str(image_path.relative_to(directory))] = sha256_file(
            image_path
        )
    return values


def write_seed_checksums(seed: int) -> None:
    values = benchmark_checksums(seed)
    with seed_sha_file(seed).open("w", encoding="utf-8") as file:
        for name, digest in values.items():
            file.write(f"{digest}  {name}\n")


def verify_seed_benchmark(seed: int) -> None:
    checksum_path = seed_sha_file(seed)
    if not checksum_path.exists():
        raise FileNotFoundError(checksum_path)
    expected = parse_sha256_file(checksum_path)
    observed = benchmark_checksums(seed)
    if expected != observed:
        raise RuntimeError(
            f"Image-seed benchmark {seed} failed SHA256 verification."
        )


def allocate_disjoint_image_chunks(
    dataset: Any,
    step56: Any,
) -> dict[int, dict[str, list[int]]]:
    indices_by_class = step56.food101_indices_by_class(dataset)
    allocation: dict[int, dict[str, list[int]]] = {
        seed: {} for seed in IMAGE_SEEDS
    }

    needed_per_class = (
        len(IMAGE_SEEDS) * IMAGES_PER_CUISINE_PER_SEED
    )

    for cuisine_index, cuisine in enumerate(CUISINES):
        class_name = step56.CUISINE_TO_FOOD101_CLASS[cuisine]
        candidates = list(indices_by_class[class_name])

        if len(candidates) < needed_per_class:
            raise RuntimeError(
                f"Food-101 class {class_name} has {len(candidates)} "
                f"images, but {needed_per_class} disjoint images are required."
            )

        rng = random.Random(
            BASE_SEED + 10000 + cuisine_index
        )
        rng.shuffle(candidates)
        selected = candidates[:needed_per_class]

        for seed_index, seed in enumerate(IMAGE_SEEDS):
            start = seed_index * IMAGES_PER_CUISINE_PER_SEED
            end = start + IMAGES_PER_CUISINE_PER_SEED
            allocation[seed][cuisine] = selected[start:end]

    return allocation


def build_seed_benchmark(
    seed: int,
    original_rows: Sequence[dict[str, Any]],
    dataset: Any,
    allocation: dict[int, dict[str, list[int]]],
    step56: Any,
    overwrite: bool,
) -> None:
    directory = seed_dir(seed)

    if overwrite and directory.exists():
        shutil.rmtree(directory)

    if (
        seed_benchmark_file(seed).exists()
        and seed_sha_file(seed).exists()
    ):
        verify_seed_benchmark(seed)
        print(
            f"[seed {seed}] benchmark exists and passed SHA256.",
            flush=True,
        )
        return

    if directory.exists():
        shutil.rmtree(directory)
    (directory / "images").mkdir(parents=True, exist_ok=True)

    rows_by_cuisine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original_rows:
        rows_by_cuisine[str(row["target_cuisine"])].append(row)

    benchmark_rows: list[dict[str, Any]] = []
    used_indices_by_class: dict[str, set[int]] = defaultdict(set)

    for cuisine_index, cuisine in enumerate(CUISINES):
        cuisine_rows = sorted(
            rows_by_cuisine[cuisine],
            key=lambda row: str(row["benchmark_id"]),
        )
        if len(cuisine_rows) != 25:
            raise RuntimeError(
                f"Expected 25 frozen cases for {cuisine}, "
                f"found {len(cuisine_rows)}."
            )

        aligned_rows = [
            row
            for row in cuisine_rows
            if row["scenario"] in (
                "aligned_clear",
                "aligned_noisy",
            )
        ]
        conflict_rows = [
            row
            for row in cuisine_rows
            if row["scenario"] == "conflict_clear"
        ]
        if len(aligned_rows) != 15 or len(conflict_rows) != 10:
            raise RuntimeError(
                f"Unexpected scenario counts for {cuisine}: "
                f"aligned={len(aligned_rows)} conflict={len(conflict_rows)}"
            )

        aligned_indices = allocation[seed][cuisine][:15]
        conflict_cuisine = CUISINES[
            (cuisine_index + 1) % len(CUISINES)
        ]
        conflict_indices = allocation[seed][conflict_cuisine][15:25]

        for source_row, image_index in zip(
            aligned_rows,
            aligned_indices,
        ):
            noisy = source_row["scenario"] == "aligned_noisy"
            benchmark_id = str(source_row["benchmark_id"])
            image_relative = Path("images") / f"{benchmark_id}.jpg"
            output_path = directory / image_relative

            step56.save_selected_image(
                dataset,
                image_index,
                output_path,
                noisy=noisy,
            )
            class_name = step56.CUISINE_TO_FOOD101_CLASS[cuisine]
            if image_index in used_indices_by_class[class_name]:
                raise RuntimeError(
                    f"Duplicate image index in seed {seed}: "
                    f"{class_name} {image_index}"
                )
            used_indices_by_class[class_name].add(image_index)

            row = dict(source_row)
            row.update({
                "robustness_image_seed": seed,
                "image_cuisine": cuisine,
                "food101_class": class_name,
                "food101_test_index": image_index,
                "image_file": str(image_relative),
                "expected_visual_relation": "aligned",
                "expected_visual_quality": (
                    "noisy" if noisy else "clear"
                ),
            })
            benchmark_rows.append(row)

        for source_row, image_index in zip(
            conflict_rows,
            conflict_indices,
        ):
            benchmark_id = str(source_row["benchmark_id"])
            image_relative = Path("images") / f"{benchmark_id}.jpg"
            output_path = directory / image_relative

            step56.save_selected_image(
                dataset,
                image_index,
                output_path,
                noisy=False,
            )
            class_name = step56.CUISINE_TO_FOOD101_CLASS[
                conflict_cuisine
            ]
            if image_index in used_indices_by_class[class_name]:
                raise RuntimeError(
                    f"Duplicate image index in seed {seed}: "
                    f"{class_name} {image_index}"
                )
            used_indices_by_class[class_name].add(image_index)

            row = dict(source_row)
            row.update({
                "robustness_image_seed": seed,
                "image_cuisine": conflict_cuisine,
                "food101_class": class_name,
                "food101_test_index": image_index,
                "image_file": str(image_relative),
                "expected_visual_relation": "conflict",
                "expected_visual_quality": "clear",
            })
            benchmark_rows.append(row)

    benchmark_rows.sort(
        key=lambda row: str(row["benchmark_id"])
    )

    if len(benchmark_rows) != 300:
        raise RuntimeError(
            f"Seed {seed} expected 300 rows, got {len(benchmark_rows)}."
        )

    write_jsonl(seed_benchmark_file(seed), benchmark_rows)
    write_seed_checksums(seed)
    verify_seed_benchmark(seed)

    scenario_counts = Counter(
        row["scenario"] for row in benchmark_rows
    )
    print(
        f"[seed {seed}] benchmark built: "
        f"{len(benchmark_rows)} rows {dict(scenario_counts)}",
        flush=True,
    )


def load_completed_predictions(
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


def generate_seed_predictions(
    seed: int,
    benchmark_rows: Sequence[dict[str, Any]],
    processor: Any,
    model: Any,
    step56: Any,
    step57: Any,
    overwrite: bool,
) -> Path:
    directory = seed_dir(seed)
    output_path = directory / "p1_focused_parser_predictions.jsonl"

    if overwrite and output_path.exists():
        output_path.unlink()

    allowed_ids = {
        str(row["benchmark_id"])
        for row in benchmark_rows
    }
    existing = load_completed_predictions(
        output_path,
        allowed_ids,
    )
    pending = [
        row
        for row in benchmark_rows
        if str(row["benchmark_id"]) not in existing
    ]

    if not pending:
        print(
            f"[seed {seed}] predictions already complete; skipping.",
            flush=True,
        )
        return output_path

    prompt = step57.focused_chat_text(processor)
    completed = len(existing)

    print(
        f"[seed {seed}] Existing={len(existing)} "
        f"Pending={len(pending)}",
        flush=True,
    )

    for benchmark_row in pending:
        benchmark_id = str(benchmark_row["benchmark_id"])
        image_path = directory / str(benchmark_row["image_file"])
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
        parsed = step57.parse_focused_json(raw_output)

        append_jsonl(
            output_path,
            {
                "benchmark_id": benchmark_id,
                "source_sample_id": benchmark_row["source_sample_id"],
                "image_seed": seed,
                "raw_output": raw_output,
                "parsed": parsed,
                "generation_seconds": elapsed,
            },
        )

        completed += 1
        print(
            f"[seed {seed}] {completed:03d}/300 {benchmark_id}",
            flush=True,
        )
        image.close()

    return output_path


def aggregate_rate(values: Sequence[bool]) -> float:
    return sum(bool(value) for value in values) / len(values)


def evaluate_seed(
    seed: int,
    benchmark_rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    t0_by_id: dict[str, dict[str, Any]],
    step56: Any,
    step57: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark_by_id = {
        str(row["benchmark_id"]): row
        for row in benchmark_rows
    }
    prediction_by_id = {
        str(row["benchmark_id"]): row
        for row in predictions
    }

    if set(benchmark_by_id) != set(prediction_by_id):
        raise RuntimeError(
            f"Seed {seed} benchmark/prediction IDs do not match."
        )

    scored: list[dict[str, Any]] = []

    for benchmark_id in sorted(benchmark_by_id):
        benchmark_row = benchmark_by_id[benchmark_id]
        prediction = prediction_by_id[benchmark_id]
        parsed = prediction["parsed"]

        parser_metrics = step57.parser_metrics(
            parsed,
            benchmark_row,
            step56,
        )
        policy = step57.policy_decision(
            parser_metrics,
            gated=True,
        )
        gate_safe = step57.expected_policy_safe(
            benchmark_row,
            parser_metrics,
            policy,
        )

        t0_row = t0_by_id[benchmark_id]
        text_metrics = t0_row["explanation_metrics"]

        scenario = str(benchmark_row["scenario"])
        if scenario == "conflict_clear":
            conflict_exact = bool(
                parser_metrics["visual_cuisine_accuracy"]
                and parser_metrics[
                    "visual_relation_safety_accuracy"
                ]
                and parser_metrics[
                    "visual_quality_safety_accuracy"
                ]
                and gate_safe
                and policy["database_priority"]
                and text_metrics["faithfulness_pass"]
            )
        else:
            conflict_exact = None

        if scenario == "aligned_clear":
            visual_component_exact = bool(
                parser_metrics["visual_cuisine_accuracy"]
                and parser_metrics[
                    "visual_relation_safety_accuracy"
                ]
                and parser_metrics[
                    "visual_quality_safety_accuracy"
                ]
                and gate_safe
            )
        elif scenario == "aligned_noisy":
            visual_component_exact = bool(
                parser_metrics[
                    "visual_relation_safety_accuracy"
                ]
                and parser_metrics[
                    "visual_quality_safety_accuracy"
                ]
                and gate_safe
            )
        else:
            visual_component_exact = bool(conflict_exact)

        multimodal_exact = bool(
            text_metrics["all_constraint_exact"]
            and visual_component_exact
        )
        database_override_error = bool(
            scenario in ("aligned_noisy", "conflict_clear")
            and not policy["database_priority"]
        )

        scored.append({
            "benchmark_id": benchmark_id,
            "image_seed": seed,
            "scenario": scenario,
            "target_cuisine": benchmark_row["target_cuisine"],
            "image_cuisine": benchmark_row["image_cuisine"],
            "parser_metrics": parser_metrics,
            "policy": policy,
            "gate_safety_accuracy": gate_safe,
            "conflict_resolution_exact": conflict_exact,
            "visual_component_exact": visual_component_exact,
            "multimodal_exact": multimodal_exact,
            "database_override_error": database_override_error,
            "text_exact": bool(
                text_metrics["all_constraint_exact"]
            ),
            "text_faithfulness": bool(
                text_metrics["faithfulness_pass"]
            ),
        })

    conflict_rows = [
        row
        for row in scored
        if row["conflict_resolution_exact"] is not None
    ]

    summary = {
        "image_seed": seed,
        "sample_count": len(scored),
        "visual_cuisine_accuracy": aggregate_rate([
            row["parser_metrics"]["visual_cuisine_accuracy"]
            for row in scored
        ]),
        "visual_relation_safety_accuracy": aggregate_rate([
            row["parser_metrics"][
                "visual_relation_safety_accuracy"
            ]
            for row in scored
        ]),
        "visual_quality_safety_accuracy": aggregate_rate([
            row["parser_metrics"][
                "visual_quality_safety_accuracy"
            ]
            for row in scored
        ]),
        "json_parse_success_rate": aggregate_rate([
            row["parser_metrics"]["json_parse_success"]
            for row in scored
        ]),
        "gate_safety_accuracy": aggregate_rate([
            row["gate_safety_accuracy"]
            for row in scored
        ]),
        "conflict_resolution_exact_accuracy": aggregate_rate([
            row["conflict_resolution_exact"]
            for row in conflict_rows
        ]),
        "database_override_error_rate": aggregate_rate([
            row["database_override_error"]
            for row in scored
        ]),
        "text_exact_accuracy": aggregate_rate([
            row["text_exact"]
            for row in scored
        ]),
        "text_faithfulness_rate": aggregate_rate([
            row["text_faithfulness"]
            for row in scored
        ]),
        "multimodal_exact_accuracy": aggregate_rate([
            row["multimodal_exact"]
            for row in scored
        ]),
    }

    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario in (
        "aligned_clear",
        "aligned_noisy",
        "conflict_clear",
    ):
        rows = [
            row for row in scored
            if row["scenario"] == scenario
        ]
        conflict_values = [
            row["conflict_resolution_exact"]
            for row in rows
            if row["conflict_resolution_exact"] is not None
        ]
        scenario_summaries[scenario] = {
            "sample_count": len(rows),
            "visual_cuisine_accuracy": aggregate_rate([
                row["parser_metrics"]["visual_cuisine_accuracy"]
                for row in rows
            ]),
            "visual_relation_safety_accuracy": aggregate_rate([
                row["parser_metrics"][
                    "visual_relation_safety_accuracy"
                ]
                for row in rows
            ]),
            "visual_quality_safety_accuracy": aggregate_rate([
                row["parser_metrics"][
                    "visual_quality_safety_accuracy"
                ]
                for row in rows
            ]),
            "gate_safety_accuracy": aggregate_rate([
                row["gate_safety_accuracy"]
                for row in rows
            ]),
            "conflict_resolution_exact_accuracy": (
                aggregate_rate(conflict_values)
                if conflict_values
                else None
            ),
            "multimodal_exact_accuracy": aggregate_rate([
                row["multimodal_exact"]
                for row in rows
            ]),
        }

    summary["scenarios"] = scenario_summaries
    return summary, scored


def bootstrap_seed_mean_ci(
    values: Sequence[float],
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    bootstrap_means: list[float] = []

    for _ in range(BOOTSTRAP_REPETITIONS):
        sample = [
            values[rng.randrange(len(values))]
            for _ in range(len(values))
        ]
        bootstrap_means.append(statistics.fmean(sample))

    bootstrap_means.sort()
    lower = bootstrap_means[
        int(0.025 * (BOOTSTRAP_REPETITIONS - 1))
    ]
    upper = bootstrap_means[
        int(0.975 * (BOOTSTRAP_REPETITIONS - 1))
    ]
    return [lower, upper]


def aggregate_across_seeds(
    seed_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "visual_cuisine_accuracy",
        "visual_relation_safety_accuracy",
        "visual_quality_safety_accuracy",
        "json_parse_success_rate",
        "gate_safety_accuracy",
        "conflict_resolution_exact_accuracy",
        "database_override_error_rate",
        "text_exact_accuracy",
        "text_faithfulness_rate",
        "multimodal_exact_accuracy",
    )

    aggregate: dict[str, Any] = {
        "seed_count": len(seed_summaries),
        "image_seeds": [
            summary["image_seed"]
            for summary in seed_summaries
        ],
        "metrics": {},
        "scenarios": {},
    }

    for metric_index, metric in enumerate(metric_names):
        values = [
            float(summary[metric])
            for summary in seed_summaries
        ]
        aggregate["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "sample_sd": (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),
            "min": min(values),
            "max": max(values),
            "bootstrap_95ci": bootstrap_seed_mean_ci(
                values,
                seed=BASE_SEED + metric_index,
            ),
            "per_seed": values,
        }

    for scenario in (
        "aligned_clear",
        "aligned_noisy",
        "conflict_clear",
    ):
        aggregate["scenarios"][scenario] = {}
        scenario_metric_names = (
            "visual_cuisine_accuracy",
            "visual_relation_safety_accuracy",
            "visual_quality_safety_accuracy",
            "gate_safety_accuracy",
            "conflict_resolution_exact_accuracy",
            "multimodal_exact_accuracy",
        )
        for metric_index, metric in enumerate(scenario_metric_names):
            values = [
                summary["scenarios"][scenario][metric]
                for summary in seed_summaries
            ]
            numeric_values = [
                float(value)
                for value in values
                if value is not None
            ]
            aggregate["scenarios"][scenario][metric] = (
                None
                if not numeric_values
                else {
                    "mean": statistics.fmean(numeric_values),
                    "sample_sd": (
                        statistics.stdev(numeric_values)
                        if len(numeric_values) > 1
                        else 0.0
                    ),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "bootstrap_95ci": bootstrap_seed_mean_ci(
                        numeric_values,
                        seed=(
                            BASE_SEED
                            + 100
                            + metric_index
                            + 10
                            * (
                                "aligned_clear",
                                "aligned_noisy",
                                "conflict_clear",
                            ).index(scenario)
                        ),
                    ),
                }
            )

    return aggregate


def robustness_decision(
    aggregate: dict[str, Any],
    step57_reference: dict[str, Any],
) -> dict[str, Any]:
    metrics = aggregate["metrics"]
    original_v3_conflict = float(
        step57_reference["methods"]["V3"][
            "conflict_resolution_exact_accuracy"
        ]
    )

    criteria = {
        "all_seeds_zero_database_override": (
            metrics["database_override_error_rate"]["max"]
            == 0.0
        ),
        "text_exact_preserved": (
            metrics["text_exact_accuracy"]["min"]
            == metrics["text_exact_accuracy"]["max"]
            == float(
                step57_reference["methods"]["T0"][
                    "all_constraint_exact_accuracy"
                ]
            )
        ),
        "text_faithfulness_preserved": (
            metrics["text_faithfulness_rate"]["min"]
            == metrics["text_faithfulness_rate"]["max"]
            == float(
                step57_reference["methods"]["T0"][
                    "faithfulness_rate"
                ]
            )
        ),
        "mean_gate_safety_above_50pct": (
            metrics["gate_safety_accuracy"]["mean"] > 0.50
        ),
        "mean_conflict_exact_above_monolithic_v3": (
            metrics[
                "conflict_resolution_exact_accuracy"
            ]["mean"]
            > original_v3_conflict
        ),
        "mean_multimodal_exact_above_20pct": (
            metrics["multimodal_exact_accuracy"]["mean"] > 0.20
        ),
    }

    passed = all(criteria.values())

    return {
        "status": (
            "robustness_validated"
            if passed
            else "report_with_limitations"
        ),
        "criteria": criteria,
        "final_vlm_architecture": (
            "Focused image-only VLM parser -> deterministic "
            "conflict/noise gate -> existing RAG+LoRA explanation"
        ),
        "important_note": (
            "This is the final frozen VLM robustness validation over "
            "eight disjoint Food-101 image seeds. No additional image-seed, "
            "prompt, confidence-threshold, or gate-parameter sweep is permitted."
        ),
    }


def write_tables(
    seed_summaries: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    per_seed_csv = OUT_ROOT / "vlm_robustness_per_seed.csv"
    main_csv = OUT_ROOT / "vlm_robustness_summary.csv"
    main_md = OUT_ROOT / "vlm_robustness_summary.md"
    scenario_csv = OUT_ROOT / "vlm_robustness_scenarios.csv"

    per_seed_headers = [
        "Image Seed",
        "Visual Cuisine",
        "Relation Safety",
        "Quality Safety",
        "JSON Parse",
        "Gate Safety",
        "Conflict Exact",
        "DB Override Error",
        "Text Exact",
        "Text Faithfulness",
        "Multimodal Exact",
    ]
    with per_seed_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=per_seed_headers)
        writer.writeheader()
        for summary in seed_summaries:
            writer.writerow({
                "Image Seed": summary["image_seed"],
                "Visual Cuisine": summary["visual_cuisine_accuracy"],
                "Relation Safety": summary[
                    "visual_relation_safety_accuracy"
                ],
                "Quality Safety": summary[
                    "visual_quality_safety_accuracy"
                ],
                "JSON Parse": summary["json_parse_success_rate"],
                "Gate Safety": summary["gate_safety_accuracy"],
                "Conflict Exact": summary[
                    "conflict_resolution_exact_accuracy"
                ],
                "DB Override Error": summary[
                    "database_override_error_rate"
                ],
                "Text Exact": summary["text_exact_accuracy"],
                "Text Faithfulness": summary[
                    "text_faithfulness_rate"
                ],
                "Multimodal Exact": summary[
                    "multimodal_exact_accuracy"
                ],
            })

    metric_labels = {
        "visual_cuisine_accuracy": "Visual Cuisine Accuracy",
        "visual_relation_safety_accuracy": "Relation Safety Accuracy",
        "visual_quality_safety_accuracy": "Quality Safety Accuracy",
        "json_parse_success_rate": "JSON Parse Success",
        "gate_safety_accuracy": "Gate Safety Accuracy",
        "conflict_resolution_exact_accuracy": (
            "Conflict Resolution Exact Accuracy"
        ),
        "database_override_error_rate": "Database Override Error Rate",
        "text_exact_accuracy": "Text Exact Accuracy",
        "text_faithfulness_rate": "Text Faithfulness",
        "multimodal_exact_accuracy": "Multimodal Exact Accuracy",
    }

    summary_headers = [
        "Metric",
        "Mean",
        "Sample SD",
        "Min",
        "Max",
        "Bootstrap 95% CI Lower",
        "Bootstrap 95% CI Upper",
    ]
    summary_rows: list[dict[str, Any]] = []

    for metric, label in metric_labels.items():
        value = aggregate["metrics"][metric]
        summary_rows.append({
            "Metric": label,
            "Mean": value["mean"],
            "Sample SD": value["sample_sd"],
            "Min": value["min"],
            "Max": value["max"],
            "Bootstrap 95% CI Lower": value[
                "bootstrap_95ci"
            ][0],
            "Bootstrap 95% CI Upper": value[
                "bootstrap_95ci"
            ][1],
        })

    with main_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown = [
        "| " + " | ".join(summary_headers) + " |",
        "| " + " | ".join(["---"] * len(summary_headers)) + " |",
    ]
    for row in summary_rows:
        markdown.append(
            "| "
            + " | ".join(
                [
                    str(row["Metric"]),
                    f"{float(row['Mean']):.4f}",
                    f"{float(row['Sample SD']):.4f}",
                    f"{float(row['Min']):.4f}",
                    f"{float(row['Max']):.4f}",
                    f"{float(row['Bootstrap 95% CI Lower']):.4f}",
                    f"{float(row['Bootstrap 95% CI Upper']):.4f}",
                ]
            )
            + " |"
        )
    main_md.write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    scenario_headers = [
        "Scenario",
        "Metric",
        "Mean",
        "Sample SD",
        "Min",
        "Max",
        "Bootstrap 95% CI Lower",
        "Bootstrap 95% CI Upper",
    ]
    with scenario_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=scenario_headers)
        writer.writeheader()

        for scenario, metrics in aggregate["scenarios"].items():
            for metric, value in metrics.items():
                if value is None:
                    continue
                writer.writerow({
                    "Scenario": scenario,
                    "Metric": metric,
                    "Mean": value["mean"],
                    "Sample SD": value["sample_sd"],
                    "Min": value["min"],
                    "Max": value["max"],
                    "Bootstrap 95% CI Lower": value[
                        "bootstrap_95ci"
                    ][0],
                    "Bootstrap 95% CI Upper": value[
                        "bootstrap_95ci"
                    ][1],
                })


def main() -> None:
    args = parse_args()

    for path in (
        STEP56_FILE,
        STEP57_FILE,
        ORIGINAL_BENCHMARK_FILE,
        T0_SCORED_FILE,
        STEP57_SUMMARY_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    verify_original_benchmark()

    step56 = load_module(
        STEP56_FILE,
        "fh_step56_robustness",
    )
    step57 = load_module(
        STEP57_FILE,
        "fh_step57_robustness",
    )

    original_rows = read_jsonl(ORIGINAL_BENCHMARK_FILE)
    if len(original_rows) != 300:
        raise RuntimeError(
            f"Expected 300 original benchmark rows, got "
            f"{len(original_rows)}."
        )

    selected_seeds = (
        IMAGE_SEEDS[: args.limit_seeds]
        if args.limit_seeds is not None
        else IMAGE_SEEDS
    )
    if not selected_seeds:
        raise ValueError("--limit-seeds must be >= 1.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment": (
            "fitness_home_modular_vlm_disjoint_image_seed_robustness_v1"
        ),
        "status": "frozen_before_result_inspection",
        "model": MODEL_ID,
        "image_seeds": list(selected_seeds),
        "full_image_seed_plan": list(IMAGE_SEEDS),
        "images_per_seed": 300,
        "images_per_cuisine_per_seed": IMAGES_PER_CUISINE_PER_SEED,
        "food101_images_are_disjoint_across_full_eight_seeds": True,
        "focused_parser_prompt": step57.FOCUSED_SYSTEM_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "architecture": (
            "Focused image-only parser -> deterministic conflict/noise "
            "gate -> existing RAG+LoRA explanation"
        ),
        "development_only": True,
        "blind_test_used": False,
        "no_parameter_sweep": True,
    }
    protocol_path = OUT_ROOT / "vlm_robustness_protocol.json"

    if protocol_path.exists():
        existing = read_json(protocol_path)
        if existing != protocol:
            if args.limit_seeds is not None:
                protocol_path = (
                    OUT_ROOT
                    / f"vlm_robustness_protocol_smoke_{len(selected_seeds)}.json"
                )
                protocol_path.write_text(
                    json.dumps(
                        protocol,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                raise RuntimeError(
                    "Existing robustness protocol differs from the full "
                    "frozen configuration."
                )
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("=" * 78)
    print("FITNESS HOME — FINAL MODULAR VLM ROBUSTNESS VALIDATION")
    print("=" * 78)
    print("Image seeds          :", selected_seeds)
    print("Images per seed      : 300")
    print(
        "Total VLM generations:",
        300 * len(selected_seeds),
    )
    print("Model                :", MODEL_ID)
    print("Food-101 root        :", FOOD101_ROOT)
    print("Development only     : YES")
    print("Blind test used      : NO")
    print("Output               :", OUT_ROOT)

    dataset = step56.load_food101()
    allocation = allocate_disjoint_image_chunks(
        dataset,
        step56,
    )

    for seed in selected_seeds:
        build_seed_benchmark(
            seed,
            original_rows,
            dataset,
            allocation,
            step56,
            overwrite=args.overwrite_benchmarks,
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)
    set_seed(BASE_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Llama-3.2 Vision inference."
        )

    processor, model = step56.load_vlm()

    prediction_paths: dict[int, Path] = {}
    for seed in selected_seeds:
        benchmark_rows = read_jsonl(seed_benchmark_file(seed))
        prediction_paths[seed] = generate_seed_predictions(
            seed,
            benchmark_rows,
            processor,
            model,
            step56,
            step57,
            overwrite=args.overwrite_predictions,
        )

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    t0_rows = read_jsonl(T0_SCORED_FILE)
    t0_by_id = {
        str(row["benchmark_id"]): row
        for row in t0_rows
    }

    seed_summaries: list[dict[str, Any]] = []
    all_scored: list[dict[str, Any]] = []

    for seed in selected_seeds:
        benchmark_rows = read_jsonl(seed_benchmark_file(seed))
        predictions = read_jsonl(prediction_paths[seed])

        summary, scored = evaluate_seed(
            seed,
            benchmark_rows,
            predictions,
            t0_by_id,
            step56,
            step57,
        )
        seed_summaries.append(summary)
        all_scored.extend(scored)

        write_jsonl(
            seed_dir(seed) / "mv1_robustness_scored.jsonl",
            scored,
        )
        (
            seed_dir(seed) / "seed_summary.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    aggregate = aggregate_across_seeds(seed_summaries)
    step57_reference = read_json(STEP57_SUMMARY_FILE)
    decision = robustness_decision(
        aggregate,
        step57_reference,
    )

    write_tables(seed_summaries, aggregate)

    write_jsonl(
        OUT_ROOT / "mv1_all_seed_scored.jsonl",
        all_scored,
    )
    (
        OUT_ROOT / "vlm_robustness_evaluation_summary.json"
    ).write_text(
        json.dumps(
            {
                "experiment": (
                    "fitness_home_modular_vlm_disjoint_image_seed_robustness_v1"
                ),
                "development_only": True,
                "blind_test_used": False,
                "seed_summaries": seed_summaries,
                "aggregate": aggregate,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        OUT_ROOT / "final_vlm_robustness_decision.json"
    ).write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_files = [
        protocol_path,
        OUT_ROOT / "vlm_robustness_per_seed.csv",
        OUT_ROOT / "vlm_robustness_summary.csv",
        OUT_ROOT / "vlm_robustness_summary.md",
        OUT_ROOT / "vlm_robustness_scenarios.csv",
        OUT_ROOT / "mv1_all_seed_scored.jsonl",
        OUT_ROOT / "vlm_robustness_evaluation_summary.json",
        OUT_ROOT / "final_vlm_robustness_decision.json",
    ]
    checksum_path = OUT_ROOT / "SHA256SUMS.txt"
    with checksum_path.open("w", encoding="utf-8") as file:
        for path in output_files:
            file.write(f"{sha256_file(path)}  {path.name}\n")

    print()
    print("=" * 78)
    print("FINAL MODULAR VLM ROBUSTNESS VALIDATION COMPLETE")
    print("=" * 78)

    for summary in seed_summaries:
        print(
            f"seed={summary['image_seed']} "
            f"Visual={summary['visual_cuisine_accuracy']:.2%} "
            f"Relation={summary['visual_relation_safety_accuracy']:.2%} "
            f"Quality={summary['visual_quality_safety_accuracy']:.2%} "
            f"Gate={summary['gate_safety_accuracy']:.2%} "
            f"Conflict={summary['conflict_resolution_exact_accuracy']:.2%} "
            f"Multimodal={summary['multimodal_exact_accuracy']:.2%}"
        )

    print()
    for metric in (
        "visual_cuisine_accuracy",
        "visual_relation_safety_accuracy",
        "visual_quality_safety_accuracy",
        "gate_safety_accuracy",
        "conflict_resolution_exact_accuracy",
        "database_override_error_rate",
        "text_exact_accuracy",
        "text_faithfulness_rate",
        "multimodal_exact_accuracy",
    ):
        value = aggregate["metrics"][metric]
        print(
            f"{metric:42s} "
            f"mean={value['mean']:.2%} "
            f"sd={value['sample_sd']:.2%} "
            f"95%CI=[{value['bootstrap_95ci'][0]:.2%}, "
            f"{value['bootstrap_95ci'][1]:.2%}]"
        )

    print("Decision status      :", decision["status"])
    print("Decision criteria    :", decision["criteria"])
    print("Summary table        :", OUT_ROOT / "vlm_robustness_summary.md")
    print(
        "Decision file        :",
        OUT_ROOT / "final_vlm_robustness_decision.json",
    )
    print("Blind test used      : NO")


if __name__ == "__main__":
    main()
