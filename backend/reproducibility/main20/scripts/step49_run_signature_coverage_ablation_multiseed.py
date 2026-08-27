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

OUT_ROOT = HERE / "34_signature_coverage_ablation"

SELECTION_SEEDS = (20260813, 20260814, 20260815)
BUDGETS = {
    "10pct": 1598,
    "5pct": 799,
}
MATCH_ORDER = ("full", "weak", "partial")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_no}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def largest_remainder_targets(
    full_match_counts: Counter[str],
    target_n: int,
) -> dict[str, int]:
    total = sum(full_match_counts.values())
    raw = {
        match: target_n * full_match_counts[match] / total
        for match in MATCH_ORDER
    }
    targets = {
        match: math.floor(raw[match])
        for match in MATCH_ORDER
    }

    remaining = target_n - sum(targets.values())
    order = sorted(
        MATCH_ORDER,
        key=lambda match: (
            raw[match] - targets[match],
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


def build_nested_no_signature_subsets(
    rows: list[dict[str, Any]],
    full_match_counts: Counter[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Distribution-matched random sampling with no constraint-signature
    coverage requirement.

    The 5% subset is nested inside the 10% subset. Full/Weak/Partial
    quotas remain matched to the source train distribution so that the
    only major ablation is removal of the all-signature coverage rule.
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

    quotas5 = largest_remainder_targets(
        full_match_counts,
        BUDGETS["5pct"],
    )
    quotas10 = largest_remainder_targets(
        full_match_counts,
        BUDGETS["10pct"],
    )

    subset5: list[dict[str, Any]] = []
    subset10: list[dict[str, Any]] = []

    for match in MATCH_ORDER:
        q5 = quotas5[match]
        q10 = quotas10[match]

        if q10 > len(pools[match]):
            raise RuntimeError(
                f"Not enough {match} samples for 10% quota."
            )

        selected10 = pools[match][:q10]
        selected5 = selected10[:q5]

        subset5.extend(selected5)
        subset10.extend(selected10)

    subset5.sort(key=lambda row: str(row["sample_id"]))
    subset10.sort(key=lambda row: str(row["sample_id"]))

    ids5 = {str(row["sample_id"]) for row in subset5}
    ids10 = {str(row["sample_id"]) for row in subset10}

    if not ids5 <= ids10:
        raise RuntimeError(
            f"Nestedness failed for selection seed {seed}"
        )

    if len(subset5) != BUDGETS["5pct"]:
        raise RuntimeError(
            f"5% count mismatch: {len(subset5)}"
        )
    if len(subset10) != BUDGETS["10pct"]:
        raise RuntimeError(
            f"10% count mismatch: {len(subset10)}"
        )

    if coverage(subset5)["match"] != quotas5:
        raise RuntimeError(
            f"5% match quota mismatch for seed {seed}"
        )
    if coverage(subset10)["match"] != quotas10:
        raise RuntimeError(
            f"10% match quota mismatch for seed {seed}"
        )

    return subset5, subset10


def patch_training_script(
    source: str,
    subset_relative_path: str,
    run_directory: str,
    train_count: int,
    experiment_name: str,
) -> str:
    text = source

    parts = subset_relative_path.split("/")
    train_expr = (
        f'TRAIN_FILE = EXPERIMENT_ROOT / "{parts[0]}"'
        + "".join(
            f' / "{part}"'
            for part in parts[1:]
        )
    )

    text, count = re.subn(
        r"^TRAIN_FILE\s*=.*$",
        train_expr,
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
            "Could not identify adapter directory in "
            "evaluation source."
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
    env: dict[str, str],
) -> None:
    print()
    print("$", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=HERE,
        env=env,
        check=True,
    )


def main() -> None:
    for path in (TRAIN_POOL, BASE_TRAINER):
        if not path.exists():
            raise FileNotFoundError(path)

    eval_source_path = next(
        (
            path
            for path in EVAL_CANDIDATES
            if path.exists()
        ),
        None,
    )
    if eval_source_path is None:
        raise FileNotFoundError(
            "No validated Main-20K evaluation script found."
        )

    train_rows = read_jsonl(TRAIN_POOL)
    if len(train_rows) != 15983:
        raise RuntimeError(
            f"Expected 15983 train samples, got {len(train_rows)}"
        )

    full_match_counts = Counter(
        str(metadata_of(row).get("match_type", ""))
        for row in train_rows
    )

    if set(full_match_counts) != set(MATCH_ORDER):
        raise RuntimeError(
            f"Unexpected match types: {dict(full_match_counts)}"
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    trainer_source = BASE_TRAINER.read_text(
        encoding="utf-8"
    )
    eval_source = eval_source_path.read_text(
        encoding="utf-8"
    )

    jobs: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "experiment": (
            "signature_coverage_ablation_"
            "distribution_matched_random"
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
            "Full/Weak/Partial distribution, and 5%-inside-10% "
            "nestedness, but deliberately do not enforce coverage "
            "of all 500 constraint signatures."
        ),
        "subsets": {},
    }

    print("=" * 72)
    print("SIGNATURE-COVERAGE ABLATION PIPELINE")
    print("=" * 72)
    print("Train pool        :", len(train_rows))
    print("Source signatures :", coverage(train_rows)["signatures"])
    print("Selection seeds   :", SELECTION_SEEDS)
    print("Budgets           :", BUDGETS)
    print("Blind test used   : False")
    print("Evaluation source :", eval_source_path.name)

    for seed in SELECTION_SEEDS:
        subset5, subset10 = (
            build_nested_no_signature_subsets(
                rows=train_rows,
                full_match_counts=full_match_counts,
                seed=seed,
            )
        )

        ids5 = {
            str(row["sample_id"])
            for row in subset5
        }
        ids10 = {
            str(row["sample_id"])
            for row in subset10
        }

        for budget, subset in (
            ("5pct", subset5),
            ("10pct", subset10),
        ):
            seed_dir = OUT_ROOT / f"seed_{seed}"
            subset_path = (
                seed_dir
                / f"train_no_signature_{budget}.jsonl"
            )
            write_jsonl(subset_path, subset)

            cov = coverage(subset)
            summary["subsets"][
                f"{budget}_seed_{seed}"
            ] = {
                **cov,
                "subset_file": str(
                    subset_path.relative_to(HERE)
                ),
                "nested_5_in_10": ids5 <= ids10,
            }

            run_directory = (
                f"35_no_signature_{budget}_seed_{seed}"
            )
            evaluation_root = (
                f"36_no_signature_{budget}_seed_{seed}_dev_eval"
            )
            experiment_name = (
                "tinyllama_qlora_main20k_"
                f"no_signature_random_{budget}_seed_{seed}"
            )

            train_script = (
                HERE
                / f"step49_train_no_signature_{budget}_"
                f"seed_{seed}.py"
            )
            eval_script = (
                HERE
                / f"step49_eval_no_signature_{budget}_"
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

            eval_script.write_text(
                patch_evaluation_script(
                    source=eval_source,
                    adapter_run_directory=run_directory,
                    evaluation_root=evaluation_root,
                ),
                encoding="utf-8",
            )

            jobs.append({
                "seed": seed,
                "budget": budget,
                "train_script": train_script,
                "eval_script": eval_script,
                "run_directory": HERE / run_directory,
                "evaluation_directory": (
                    HERE / evaluation_root
                ),
            })

    summary_path = (
        OUT_ROOT
        / "signature_coverage_ablation_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault(
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
            env=env,
        )
        run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(job["eval_script"]),
            ],
            env=env,
        )

    jobs.sort(
        key=lambda job: (
            0 if job["budget"] == "10pct" else 1,
            job["seed"],
        )
    )

    result_paths: list[str] = []

    for index, job in enumerate(jobs, 1):
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
            f"NO-SIGNATURE RANDOM {budget} "
            f"SEED {seed}"
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
                env=env,
            )
            run(
                [
                    sys.executable,
                    str(job["train_script"]),
                    "--overwrite",
                ],
                env=env,
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
                    str(job["eval_script"]),
                    "--split",
                    "test",
                    "--batch-size",
                    "8",
                    "--overwrite",
                ],
                env=env,
            )
        else:
            print(
                "Evaluation already complete; skipping."
            )

        result_paths.append(
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
                "evaluation_summaries": result_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "SIGNATURE-COVERAGE ABLATION PIPELINE COMPLETE"
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
