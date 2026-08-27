#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

TRAIN_POOL = HERE / "04_main20k_split" / "train.jsonl"
BASE_TRAINER = HERE / "step35_train_main20k_matched_random_50pct.py"

EVAL_CANDIDATES = [
    HERE / "step45_evaluate_megd_50pct_dev.py",
    HERE / "step36_evaluate_matched_random_50pct.py",
    HERE / "step31_evaluate_bcaegd_50pct.py",
]

OUT_ROOT = HERE / "37_ultra_low_budget_random_multiseed"

SELECTION_SEEDS = (20260813, 20260814, 20260815)
BUDGETS = {
    "2p5pct": 400,
    "1pct": 160,
}
MATCH_ORDER = ("full", "weak", "partial")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def largest_remainder_targets(
    full_match_counts: Counter[str],
    target_size: int,
) -> dict[str, int]:
    total = sum(full_match_counts.values())
    exact = {
        match: target_size * full_match_counts[match] / total
        for match in MATCH_ORDER
    }
    targets = {
        match: math.floor(exact[match])
        for match in MATCH_ORDER
    }

    remaining = target_size - sum(targets.values())
    order = sorted(
        MATCH_ORDER,
        key=lambda match: (
            exact[match] - targets[match],
            match,
        ),
        reverse=True,
    )

    for match in order[:remaining]:
        targets[match] += 1

    return targets


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "signatures": len({
            str(metadata_of(row).get("constraint_signature_id", ""))
            for row in rows
        } - {""}),
        "queries": len({
            str(metadata_of(row).get("query_id", ""))
            for row in rows
        } - {""}),
        "restaurants": len({
            str(metadata_of(row).get("restaurant_name", ""))
            for row in rows
        } - {""}),
        "match": dict(Counter(
            str(metadata_of(row).get("match_type", ""))
            for row in rows
        )),
    }


def build_nested_subsets(
    rows: list[dict[str, Any]],
    full_match_counts: Counter[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Distribution-matched random sampling with no signature-coverage rule.

    The 1% subset is nested inside the 2.5% subset. Full/Weak/Partial
    proportions remain matched to the source training distribution.
    """
    rng = random.Random(seed)

    pools: dict[str, list[dict[str, Any]]] = {}
    for match in MATCH_ORDER:
        pool = [
            row
            for row in rows
            if str(metadata_of(row).get("match_type", "")) == match
        ]
        rng.shuffle(pool)
        pools[match] = pool

    quota_1 = largest_remainder_targets(
        full_match_counts,
        BUDGETS["1pct"],
    )
    quota_2p5 = largest_remainder_targets(
        full_match_counts,
        BUDGETS["2p5pct"],
    )

    subset_1: list[dict[str, Any]] = []
    subset_2p5: list[dict[str, Any]] = []

    for match in MATCH_ORDER:
        n_1 = quota_1[match]
        n_2p5 = quota_2p5[match]

        if n_2p5 > len(pools[match]):
            raise RuntimeError(
                f"Not enough {match} samples for 2.5% quota."
            )

        chosen_2p5 = pools[match][:n_2p5]
        chosen_1 = chosen_2p5[:n_1]

        subset_1.extend(chosen_1)
        subset_2p5.extend(chosen_2p5)

    subset_1.sort(key=lambda row: str(row["sample_id"]))
    subset_2p5.sort(key=lambda row: str(row["sample_id"]))

    ids_1 = {str(row["sample_id"]) for row in subset_1}
    ids_2p5 = {str(row["sample_id"]) for row in subset_2p5}

    if not ids_1 <= ids_2p5:
        raise RuntimeError(
            f"Nestedness failed for selection seed {seed}"
        )

    if len(subset_1) != BUDGETS["1pct"]:
        raise RuntimeError(
            f"1% count mismatch: {len(subset_1)}"
        )
    if len(subset_2p5) != BUDGETS["2p5pct"]:
        raise RuntimeError(
            f"2.5% count mismatch: {len(subset_2p5)}"
        )

    if coverage(subset_1)["match"] != quota_1:
        raise RuntimeError(
            f"1% match quota mismatch for seed {seed}"
        )
    if coverage(subset_2p5)["match"] != quota_2p5:
        raise RuntimeError(
            f"2.5% match quota mismatch for seed {seed}"
        )

    return subset_1, subset_2p5


def patch_training_script(
    source: str,
    subset_relative_path: str,
    run_directory: str,
    train_count: int,
    experiment_name: str,
) -> str:
    text = source

    parts = subset_relative_path.split("/")
    train_expression = (
        f'TRAIN_FILE = EXPERIMENT_ROOT / "{parts[0]}"'
        + "".join(
            f' / "{part}"'
            for part in parts[1:]
        )
    )

    text, count = re.subn(
        r"^TRAIN_FILE\s*=.*$",
        train_expression,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Could not patch TRAIN_FILE")

    text, count = re.subn(
        r"^RUN_ROOT\s*=.*$",
        f'RUN_ROOT = EXPERIMENT_ROOT / "{run_directory}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Could not patch RUN_ROOT")

    text, count = re.subn(
        r"^EXPECTED_TRAIN_SAMPLES\s*=\s*\d+\s*$",
        f"EXPECTED_TRAIN_SAMPLES = {train_count}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(
            "Could not patch EXPECTED_TRAIN_SAMPLES"
        )

    expected_total = train_count + 1948 + 2069
    text, count = re.subn(
        r"^EXPECTED_TOTAL_SAMPLES\s*=\s*\d+\s*$",
        f"EXPECTED_TOTAL_SAMPLES = {expected_total}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(
            "Could not patch EXPECTED_TOTAL_SAMPLES"
        )

    text, count = re.subn(
        r'"experiment"\s*:\s*"[^"]+"',
        f'"experiment": "{experiment_name}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(
            "Could not patch experiment name"
        )

    return text


def patch_evaluation_script(
    source: str,
    adapter_run_directory: str,
    evaluation_root: str,
) -> str:
    text = source

    adapter_candidates = (
        "24_main20k_qlora_megd_50pct",
        "15_main20k_qlora_matched_random_50pct",
        "10_main20k_qlora_bcaegd_50pct",
    )

    replaced_adapter = False
    for old in adapter_candidates:
        if old in text:
            text = text.replace(old, adapter_run_directory)
            replaced_adapter = True
            break

    if not replaced_adapter:
        raise RuntimeError(
            "Could not identify adapter directory "
            "in evaluation source."
        )

    output_candidates = (
        "26_megd_50pct_dev_eval",
        "16_matched_random_50pct_eval",
        "11_bcaegd_50pct_eval",
    )

    replaced_output = False
    for old in output_candidates:
        if old in text:
            text = text.replace(old, evaluation_root)
            replaced_output = True
            break

    if not replaced_output:
        raise RuntimeError(
            "Could not identify evaluation output directory."
        )

    text = text.replace(
        'metadata_of(record).get("accepted")',
        'metadata_of(record).get("filter_v2_3_accepted")',
    )

    return text


def run(
    command: list[str],
    environment: dict[str, str],
) -> None:
    print()
    print("$", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=HERE,
        env=environment,
        check=True,
    )


def main() -> None:
    for path in (TRAIN_POOL, BASE_TRAINER):
        if not path.exists():
            raise FileNotFoundError(path)

    evaluation_source_path = next(
        (
            path
            for path in EVAL_CANDIDATES
            if path.exists()
        ),
        None,
    )
    if evaluation_source_path is None:
        raise FileNotFoundError(
            "No validated Main-20K evaluation script found."
        )

    training_rows = read_jsonl(TRAIN_POOL)
    if len(training_rows) != 15983:
        raise RuntimeError(
            f"Expected 15983 train samples, got {len(training_rows)}"
        )

    full_match_counts = Counter(
        str(metadata_of(row).get("match_type", ""))
        for row in training_rows
    )

    if set(full_match_counts) != set(MATCH_ORDER):
        raise RuntimeError(
            f"Unexpected match types: {dict(full_match_counts)}"
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    trainer_source = BASE_TRAINER.read_text(
        encoding="utf-8"
    )
    evaluation_source = evaluation_source_path.read_text(
        encoding="utf-8"
    )

    jobs: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "experiment": (
            "ultra_low_budget_distribution_matched_random"
        ),
        "development_only": True,
        "blind_test_used": False,
        "selection_seeds": list(SELECTION_SEEDS),
        "budgets": BUDGETS,
        "source_match_distribution": dict(
            full_match_counts
        ),
        "definition": (
            "Random subsets preserve sample count, "
            "Full/Weak/Partial distribution, and 1%-inside-2.5% "
            "nestedness. No signature-coverage rule is imposed."
        ),
        "subsets": {},
    }

    print("=" * 72)
    print("ULTRA-LOW-BUDGET RANDOM MULTI-SEED PIPELINE")
    print("=" * 72)
    print("Train pool       :", len(training_rows))
    print("Source signatures:", coverage(training_rows)["signatures"])
    print("Selection seeds  :", SELECTION_SEEDS)
    print("Budgets          :", BUDGETS)
    print("Blind test used  : False")
    print("Evaluation source:", evaluation_source_path.name)

    for seed in SELECTION_SEEDS:
        subset_1, subset_2p5 = build_nested_subsets(
            rows=training_rows,
            full_match_counts=full_match_counts,
            seed=seed,
        )

        ids_1 = {
            str(row["sample_id"])
            for row in subset_1
        }
        ids_2p5 = {
            str(row["sample_id"])
            for row in subset_2p5
        }

        for budget, subset in (
            ("1pct", subset_1),
            ("2p5pct", subset_2p5),
        ):
            seed_directory = OUT_ROOT / f"seed_{seed}"
            subset_path = (
                seed_directory
                / f"train_random_{budget}.jsonl"
            )
            write_jsonl(subset_path, subset)

            subset_coverage = coverage(subset)
            summary["subsets"][
                f"{budget}_seed_{seed}"
            ] = {
                **subset_coverage,
                "subset_file": str(
                    subset_path.relative_to(HERE)
                ),
                "nested_1_in_2p5": ids_1 <= ids_2p5,
            }

            run_directory = (
                f"38_random_{budget}_seed_{seed}"
            )
            evaluation_root = (
                f"39_random_{budget}_seed_{seed}_dev_eval"
            )
            experiment_name = (
                "tinyllama_qlora_main20k_"
                f"random_{budget}_seed_{seed}"
            )

            train_script = (
                HERE
                / f"step50_train_random_{budget}_"
                f"seed_{seed}.py"
            )
            evaluation_script = (
                HERE
                / f"step50_eval_random_{budget}_"
                f"seed_{seed}.py"
            )

            train_script.write_text(
                patch_training_script(
                    source=trainer_source,
                    subset_relative_path=str(
                        subset_path.relative_to(HERE)
                    ),
                    run_directory=run_directory,
                    train_count=len(subset),
                    experiment_name=experiment_name,
                ),
                encoding="utf-8",
            )

            evaluation_script.write_text(
                patch_evaluation_script(
                    source=evaluation_source,
                    adapter_run_directory=run_directory,
                    evaluation_root=evaluation_root,
                ),
                encoding="utf-8",
            )

            jobs.append({
                "seed": seed,
                "budget": budget,
                "train_script": train_script,
                "evaluation_script": evaluation_script,
                "run_directory": HERE / run_directory,
                "evaluation_directory": (
                    HERE / evaluation_root
                ),
            })

    summary_path = (
        OUT_ROOT
        / "ultra_low_budget_subset_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment.setdefault(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface"),
    )

    for job in jobs:
        run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(job["train_script"]),
            ],
            environment=environment,
        )
        run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(job["evaluation_script"]),
            ],
            environment=environment,
        )

    jobs.sort(
        key=lambda job: (
            0 if job["budget"] == "2p5pct" else 1,
            job["seed"],
        )
    )

    evaluation_paths: list[str] = []

    for index, job in enumerate(jobs, start=1):
        seed = job["seed"]
        budget = job["budget"]
        run_directory: Path = job["run_directory"]
        evaluation_directory: Path = (
            job["evaluation_directory"]
        )

        final_adapter = (
            run_directory
            / "full_run"
            / "final_adapter"
        )
        frozen_directory = (
            run_directory
            / "full_run_frozen"
        )
        evaluation_summary = (
            evaluation_directory
            / "test_final_2069"
            / "evaluation_summary.json"
        )

        print()
        print("=" * 72)
        print(
            f"JOB {index}/{len(jobs)} — "
            f"RANDOM {budget} SEED {seed}"
        )
        print("=" * 72)

        if not final_adapter.exists():
            run(
                [
                    sys.executable,
                    str(job["train_script"]),
                    "--smoke-test",
                    "--overwrite",
                ],
                environment=environment,
            )
            run(
                [
                    sys.executable,
                    str(job["train_script"]),
                    "--overwrite",
                ],
                environment=environment,
            )
        else:
            print(
                "Training already complete; skipping."
            )

        if not frozen_directory.exists():
            shutil.copytree(
                run_directory / "full_run",
                frozen_directory,
            )
        else:
            print(
                "Frozen model already exists; "
                "skipping copy."
            )

        if not evaluation_summary.exists():
            run(
                [
                    sys.executable,
                    str(job["evaluation_script"]),
                    "--split",
                    "test",
                    "--batch-size",
                    "8",
                    "--overwrite",
                ],
                environment=environment,
            )
        else:
            print(
                "Evaluation already complete; skipping."
            )

        evaluation_paths.append(
            str(evaluation_summary.relative_to(HERE))
        )

    (
        OUT_ROOT
        / "evaluation_result_paths.json"
    ).write_text(
        json.dumps(
            {
                "development_only": True,
                "blind_test_used": False,
                "evaluation_summaries": evaluation_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "ULTRA-LOW-BUDGET RANDOM PIPELINE COMPLETE"
    )
    print("=" * 72)
    print("Blind test used: False")
    print("Subset summary:", summary_path)
    print(
        "Evaluation paths:",
        OUT_ROOT / "evaluation_result_paths.json",
    )


if __name__ == "__main__":
    main()
