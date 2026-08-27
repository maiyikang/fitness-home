#!/usr/bin/env python3
from __future__ import annotations

import bisect
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

TRAIN_POOL = HERE / "04_main20k_split" / "train.jsonl"
LOSS_PROFILE = (
    HERE
    / "22_base_train_loss_profile_frozen"
    / "base_train_loss_profile.jsonl"
)
STRUCTURAL_PRIORS = (
    HERE
    / "21_hard_dev_profile"
    / "structural_error_priors.json"
)
HARD_DEV_PROFILE = (
    HERE
    / "21_hard_dev_profile"
    / "hard_development_challenge.jsonl"
)

BASE_TRAINER = HERE / "step35_train_main20k_matched_random_50pct.py"
EVAL_CANDIDATES = [
    HERE / "step45_evaluate_megd_50pct_dev.py",
    HERE / "step36_evaluate_matched_random_50pct.py",
    HERE / "step31_evaluate_bcaegd_50pct.py",
]

OUT_ROOT = HERE / "31_rmegd_v2_low_budget_multiseed"
SCRIPT_ROOT = HERE

SEEDS = (20260813, 20260814, 20260815)
BUDGETS = {
    "10pct": 1598,
    "5pct": 799,
}

FULL_MATCH_COUNTS = {
    "full": 5408,
    "weak": 5608,
    "partial": 4967,
}
MATCH_ORDER = ("full", "weak", "partial")

SMOOTHING_STRENGTH = 50.0
BOUNDARY_FALLBACK = 0.102

AVG_PATTERNS = {
    "calories": re.compile(
        r"- Average calories:\s*([0-9]+(?:\.[0-9]+)?)\s*kcal",
        re.I,
    ),
    "protein": re.compile(
        r"- Average protein:\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        re.I,
    ),
    "fiber": re.compile(
        r"- Average (?:fibre|fiber):\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        re.I,
    ),
}


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


def md(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def stable_noise(seed: int, sample_id: str) -> float:
    digest = hashlib.sha256(
        f"{seed}:{sample_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def largest_remainder_targets(target_n: int) -> dict[str, int]:
    total = sum(FULL_MATCH_COUNTS.values())
    raw = {
        key: target_n * value / total
        for key, value in FULL_MATCH_COUNTS.items()
    }
    targets = {
        key: math.floor(value)
        for key, value in raw.items()
    }
    remaining = target_n - sum(targets.values())
    order = sorted(
        raw,
        key=lambda key: (
            raw[key] - targets[key],
            key,
        ),
        reverse=True,
    )
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def infer_boundary_threshold(
    dev_rows: list[dict[str, Any]],
) -> float:
    near_values: list[float] = []
    non_near_values: list[float] = []

    for row in dev_rows:
        sf = row.get("structural_features", {})
        margin = sf.get("minimum_relative_margin")
        near = sf.get("near_boundary")

        if not isinstance(margin, (int, float)):
            continue
        if not isinstance(near, bool):
            continue

        if near:
            near_values.append(float(margin))
        else:
            non_near_values.append(float(margin))

    if near_values and non_near_values:
        max_near = max(near_values)
        min_non_near = min(non_near_values)

        if max_near < min_non_near:
            return (max_near + min_non_near) / 2.0

    return BOUNDARY_FALLBACK


def parse_average_values(input_text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, pattern in AVG_PATTERNS.items():
        match = pattern.search(input_text)
        if match:
            values[name] = float(match.group(1))
    return values


def relative_margin(value: float, threshold: float) -> float:
    denominator = max(abs(float(threshold)), 1e-12)
    return abs(float(value) - float(threshold)) / denominator


def build_structural_bin(
    row: dict[str, Any],
    boundary_threshold: float,
) -> tuple[str, dict[str, Any]]:
    metadata = md(row)
    constraints = metadata["constraints"]
    raw_checks = metadata["constraint_checks"]
    match_type = str(metadata["match_type"]).lower()

    checks = {
        ("fiber" if key == "fibre" else key): bool(value)
        for key, value in raw_checks.items()
    }

    check_names = ["cuisine", "calories", "protein"]
    if (
        constraints.get("min_fiber") is not None
        or "fiber" in checks
    ):
        check_names.append("fiber")

    missing = [
        name
        for name in check_names
        if name not in checks
    ]
    if missing:
        raise RuntimeError(
            f"Missing checks {missing} in {row.get('sample_id')}"
        )

    failed = [
        name
        for name in check_names
        if not checks[name]
    ]
    failed_count = len(failed)

    if failed_count == 0:
        failed_bucket = "fail0"
    elif failed_count == 1:
        failed_bucket = "fail1"
    else:
        failed_bucket = "fail2plus"

    cuisine_mismatch = not checks["cuisine"]

    averages = parse_average_values(str(row.get("input", "")))
    numeric_thresholds = {
        "calories": constraints.get("max_calories"),
        "protein": constraints.get("min_protein"),
    }
    if constraints.get("min_fiber") is not None:
        numeric_thresholds["fiber"] = constraints.get("min_fiber")

    numeric_margins: dict[str, float] = {}
    for name, threshold in numeric_thresholds.items():
        if threshold is None:
            continue
        if name not in averages:
            raise RuntimeError(
                f"Cannot parse Average {name} in "
                f"{row.get('sample_id')}"
            )
        numeric_margins[name] = relative_margin(
            averages[name],
            float(threshold),
        )

    minimum_margin = min(numeric_margins.values())
    near_boundary = minimum_margin <= boundary_threshold

    structural_bin = (
        f"{match_type}|{failed_bucket}|"
        f"{'boundary' if near_boundary else 'nonboundary'}|"
        f"{'cuisine_mismatch' if cuisine_mismatch else 'cuisine_match'}"
    )

    return structural_bin, {
        "failed_constraints": failed,
        "failed_constraint_count": failed_count,
        "cuisine_mismatch": cuisine_mismatch,
        "near_boundary": near_boundary,
        "minimum_relative_margin": minimum_margin,
        "structural_bin": structural_bin,
    }


def build_smoothed_priors(
    prior_obj: dict[str, Any],
) -> tuple[dict[str, float], float]:
    priors = prior_obj["structural_priors"]
    total_samples = sum(
        int(value["samples"])
        for value in priors.values()
    )
    total_errors = sum(
        int(value["high_confidence_errors"])
        for value in priors.values()
    )
    global_rate = total_errors / total_samples

    smoothed: dict[str, float] = {}
    for key, value in priors.items():
        n = int(value["samples"])
        errors = int(value["high_confidence_errors"])
        smoothed[key] = (
            errors + SMOOTHING_STRENGTH * global_rate
        ) / (n + SMOOTHING_STRENGTH)

    return smoothed, global_rate


def minmax(
    values: dict[str, float],
) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 0.5 for key in values}
    return {
        key: (value - low) / (high - low)
        for key, value in values.items()
    }


def build_scored_items(
    train_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    smoothed_priors: dict[str, float],
    boundary_threshold: float,
    seed: int,
) -> list[dict[str, Any]]:
    loss_by_id = {
        str(row["sample_id"]): float(
            row["base_teacher_forced_loss"]
        )
        for row in loss_rows
    }

    by_match_losses: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        match = str(md(row).get("match_type", "")).lower()
        by_match_losses[match].append(
            loss_by_id[str(row["sample_id"])]
        )

    sorted_losses = {
        match: sorted(values)
        for match, values in by_match_losses.items()
    }

    restaurant_frequency = Counter(
        str(md(row).get("restaurant_name", ""))
        for row in train_rows
    )
    query_frequency = Counter(
        str(md(row).get("query_id", ""))
        for row in train_rows
    )

    prior_normalized = minmax(smoothed_priors)

    max_restaurant_frequency = max(
        restaurant_frequency.values()
    )
    max_query_frequency = max(
        query_frequency.values()
    )

    items: list[dict[str, Any]] = []

    for row in train_rows:
        metadata = md(row)
        sample_id = str(row["sample_id"])
        match = str(metadata.get("match_type", "")).lower()
        loss = loss_by_id[sample_id]
        losses = sorted_losses[match]

        percentile = (
            bisect.bisect_right(losses, loss)
            / len(losses)
        )

        # Avoid the prior MEGD-v1 failure mode of selecting only the
        # extreme high-loss tail. This utility peaks around the
        # moderately hard 70th percentile, while still retaining a
        # smaller monotonic difficulty component.
        moderate_hard = math.exp(
            -0.5 * ((percentile - 0.70) / 0.20) ** 2
        )
        difficulty_utility = (
            0.70 * moderate_hard
            + 0.30 * percentile
        )

        structural_bin, features = build_structural_bin(
            row,
            boundary_threshold,
        )
        if structural_bin not in smoothed_priors:
            raise RuntimeError(
                "Training structural bin is absent from frozen "
                f"Development priors: {structural_bin}"
            )

        restaurant = str(
            metadata.get("restaurant_name", "")
        )
        query_id = str(metadata.get("query_id", ""))

        restaurant_rarity = 1.0 - (
            restaurant_frequency[restaurant]
            / max_restaurant_frequency
        )
        query_rarity = 1.0 - (
            query_frequency[query_id]
            / max_query_frequency
        )
        rarity_utility = (
            0.70 * restaurant_rarity
            + 0.30 * query_rarity
        )

        error_prior = smoothed_priors[structural_bin]
        error_prior_utility = prior_normalized[structural_bin]

        score = (
            0.55 * difficulty_utility
            + 0.30 * error_prior_utility
            + 0.15 * rarity_utility
            + 0.015 * stable_noise(seed, sample_id)
        )

        # Representative anchor utility: prefer a central example
        # within each signature, then use error prior and rarity as
        # small tie-break signals.
        centrality = max(
            0.0,
            1.0 - abs(percentile - 0.50) / 0.50,
        )
        anchor_score = (
            0.75 * centrality
            + 0.15 * error_prior_utility
            + 0.10 * rarity_utility
            + 0.015 * stable_noise(seed + 1, sample_id)
        )

        items.append({
            "row": row,
            "sample_id": sample_id,
            "query_id": query_id,
            "signature_id": str(
                metadata.get("constraint_signature_id", "")
            ),
            "restaurant_name": restaurant,
            "match_type": match,
            "loss": loss,
            "loss_percentile": percentile,
            "difficulty_utility": difficulty_utility,
            "error_prior": error_prior,
            "error_prior_utility": error_prior_utility,
            "rarity_utility": rarity_utility,
            "score": score,
            "anchor_score": anchor_score,
            **features,
        })

    return items


def choose_signature_anchors(
    items: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
    target_n: int,
) -> list[dict[str, Any]]:
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_signature[item["signature_id"]].append(item)

    signatures = sorted(by_signature)

    for attempt in range(500):
        rng = random.Random(seed + attempt * 100003)
        current = Counter()
        selected: list[dict[str, Any]] = []
        restaurant_counts = Counter()
        anchor_restaurant_caps = {
            name: max(4, math.ceil(cap * 1.15))
            for name, cap in restaurant_caps(items, target_n).items()
        }

        decorated = []
        for signature in signatures:
            available_match_types = {
                item["match_type"]
                for item in by_signature[signature]
            }
            decorated.append((
                len(available_match_types),
                rng.random(),
                signature,
            ))
        decorated.sort()

        failed = False

        for _, _, signature in decorated:
            candidates = [
                item
                for item in by_signature[signature]
                if current[item["match_type"]]
                < quotas[item["match_type"]]
                and restaurant_counts[item["restaurant_name"]]
                < anchor_restaurant_caps[item["restaurant_name"]]
            ]
            if not candidates:
                failed = True
                break

            # Preserve quota feasibility, then choose the most
            # representative anchor rather than the hardest sample.
            remaining_capacity = {
                match: quotas[match] - current[match]
                for match in MATCH_ORDER
            }
            max_capacity = max(
                remaining_capacity[item["match_type"]]
                for item in candidates
            )
            candidates = [
                item
                for item in candidates
                if remaining_capacity[item["match_type"]]
                == max_capacity
            ]
            chosen = max(
                candidates,
                key=lambda item: (
                    item["anchor_score"],
                    stable_noise(
                        seed + attempt,
                        item["sample_id"],
                    ),
                ),
            )
            selected.append(chosen)
            current[chosen["match_type"]] += 1
            restaurant_counts[chosen["restaurant_name"]] += 1

        if not failed:
            return selected

    raise RuntimeError(
        "Could not build one representative anchor per signature "
        "within fixed match quotas."
    )


def restaurant_caps(
    items: list[dict[str, Any]],
    target_n: int,
) -> dict[str, int]:
    full_counts = Counter(
        item["restaurant_name"]
        for item in items
    )
    total = len(items)

    caps: dict[str, int] = {}
    for restaurant, count in full_counts.items():
        expected = target_n * count / total
        caps[restaurant] = max(
            3,
            math.ceil(expected * 1.35) + 3,
        )
    return caps


def fill_subset(
    items: list[dict[str, Any]],
    initial: list[dict[str, Any]],
    quotas: dict[str, int],
    target_n: int,
    signature_cap: int,
    query_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = list(initial)
    selected_ids = {
        item["sample_id"]
        for item in selected
    }

    match_counts = Counter(
        item["match_type"]
        for item in selected
    )
    signature_counts = Counter(
        item["signature_id"]
        for item in selected
    )
    query_counts = Counter(
        item["query_id"]
        for item in selected
    )
    restaurant_counts = Counter(
        item["restaurant_name"]
        for item in selected
    )

    caps = restaurant_caps(items, target_n)

    for match in MATCH_ORDER:
        need = quotas[match] - match_counts[match]
        candidates = [
            item
            for item in items
            if item["match_type"] == match
            and item["sample_id"] not in selected_ids
        ]
        candidates.sort(
            key=lambda item: (
                -item["score"],
                -stable_noise(seed, item["sample_id"]),
            )
        )

        chosen_count = 0
        for item in candidates:
            if chosen_count >= need:
                break
            if signature_counts[item["signature_id"]] >= signature_cap:
                continue
            if query_counts[item["query_id"]] >= query_cap:
                continue
            if (
                restaurant_counts[item["restaurant_name"]]
                >= caps[item["restaurant_name"]]
            ):
                continue

            selected.append(item)
            selected_ids.add(item["sample_id"])
            match_counts[match] += 1
            signature_counts[item["signature_id"]] += 1
            query_counts[item["query_id"]] += 1
            restaurant_counts[item["restaurant_name"]] += 1
            chosen_count += 1

        if chosen_count != need:
            raise RuntimeError(
                f"R-MEGD selection could not fill {match} quota "
                f"under frozen diversity caps: selected={chosen_count}, "
                f"required={need}"
            )

    if len(selected) != target_n:
        raise RuntimeError(
            f"Subset count mismatch: {len(selected)} != {target_n}"
        )

    selected.sort(key=lambda item: item["sample_id"])
    return selected


def coverage(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "samples": len(selected),
        "signatures": len({
            item["signature_id"]
            for item in selected
        }),
        "queries": len({
            item["query_id"]
            for item in selected
        }),
        "restaurants": len({
            item["restaurant_name"]
            for item in selected
        }),
        "match": dict(Counter(
            item["match_type"]
            for item in selected
        )),
        "mean_loss_percentile": statistics.fmean(
            item["loss_percentile"]
            for item in selected
        ),
        "mean_error_prior": statistics.fmean(
            item["error_prior"]
            for item in selected
        ),
        "mean_selection_score": statistics.fmean(
            item["score"]
            for item in selected
        ),
    }


def patch_training_script(
    source: str,
    subset_rel: str,
    run_dir: str,
    train_count: int,
    experiment_name: str,
    training_seed: int,
) -> str:
    text = source

    parts = subset_rel.split("/")
    replacement = (
        f'TRAIN_FILE = EXPERIMENT_ROOT / "{parts[0]}"'
        + "".join(
            f' / "{part}"'
            for part in parts[1:]
        )
    )
    text, count = re.subn(
        r"^TRAIN_FILE\s*=.*$",
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Could not patch TRAIN_FILE")

    text, count = re.subn(
        r"^RUN_ROOT\s*=.*$",
        f'RUN_ROOT = EXPERIMENT_ROOT / "{run_dir}"',
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

    total = train_count + 1948 + 2069
    text, count = re.subn(
        r"^EXPECTED_TOTAL_SAMPLES\s*=\s*\d+\s*$",
        f"EXPECTED_TOTAL_SAMPLES = {total}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(
            "Could not patch EXPECTED_TOTAL_SAMPLES"
        )

    text, count = re.subn(
        r"^SEED\s*=\s*\d+\s*$",
        f"SEED = {training_seed}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Could not patch SEED")

    text, count = re.subn(
        r'"experiment"\s*:\s*"[^"]+"',
        f'"experiment": "{experiment_name}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not patch experiment name")

    return text


def patch_eval_script(
    source: str,
    adapter_run_dir: str,
    eval_root: str,
) -> str:
    text = source

    adapter_candidates = [
        "24_main20k_qlora_megd_50pct",
        "15_main20k_qlora_matched_random_50pct",
        "10_main20k_qlora_bcaegd_50pct",
    ]
    replaced_adapter = False
    for old in adapter_candidates:
        if old in text:
            text = text.replace(old, adapter_run_dir)
            replaced_adapter = True
            break
    if not replaced_adapter:
        raise RuntimeError(
            "Could not identify adapter directory in eval script."
        )

    eval_candidates = [
        "26_megd_50pct_dev_eval",
        "16_matched_random_50pct_eval",
        "11_bcaegd_50pct_eval",
    ]
    replaced_eval = False
    for old in eval_candidates:
        if old in text:
            text = text.replace(old, eval_root)
            replaced_eval = True
            break
    if not replaced_eval:
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
    required = [
        TRAIN_POOL,
        LOSS_PROFILE,
        STRUCTURAL_PRIORS,
        HARD_DEV_PROFILE,
        BASE_TRAINER,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    eval_source_path = next(
        (path for path in EVAL_CANDIDATES if path.exists()),
        None,
    )
    if eval_source_path is None:
        raise FileNotFoundError(
            "No validated Main-20K evaluation script found."
        )

    train_rows = read_jsonl(TRAIN_POOL)
    loss_rows = read_jsonl(LOSS_PROFILE)
    dev_rows = read_jsonl(HARD_DEV_PROFILE)

    if len(train_rows) != 15983:
        raise RuntimeError(
            f"Expected 15983 train samples, got {len(train_rows)}"
        )
    if len(loss_rows) != len(train_rows):
        raise RuntimeError(
            "Loss profile and train pool have different sizes."
        )

    train_ids = {
        str(row["sample_id"])
        for row in train_rows
    }
    loss_ids = {
        str(row["sample_id"])
        for row in loss_rows
    }
    if train_ids != loss_ids:
        raise RuntimeError(
            "Loss profile sample IDs do not match train pool."
        )

    with STRUCTURAL_PRIORS.open(
        "r",
        encoding="utf-8",
    ) as f:
        prior_obj = json.load(f)

    boundary_threshold = infer_boundary_threshold(
        dev_rows
    )
    smoothed_priors, global_error_rate = (
        build_smoothed_priors(prior_obj)
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    trainer_source = BASE_TRAINER.read_text(
        encoding="utf-8"
    )
    eval_source = eval_source_path.read_text(
        encoding="utf-8"
    )

    jobs: list[dict[str, Any]] = []
    master_summary: dict[str, Any] = {
        "experiment": "R-MEGD-v2 low-budget multi-seed",
        "status": (
            "Development benchmark method-selection only; "
            "final blind test not used"
        ),
        "blind_test_used": False,
        "seeds": list(SEEDS),
        "budgets": BUDGETS,
        "global_dev_high_confidence_error_rate": (
            global_error_rate
        ),
        "boundary_threshold": boundary_threshold,
        "method": {
            "signature_anchor": (
                "one representative near-median-loss anchor "
                "per constraint signature"
            ),
            "additional_selection": (
                "moderate-hard model difficulty + smoothed "
                "Development error prior + rarity utility"
            ),
            "anti_collapse_controls": (
                "fixed match quotas, signature caps, query caps, "
                "and source-proportional restaurant caps"
            ),
            "nested": "5pct is a strict subset of 10pct",
        },
        "subsets": {},
    }

    print("=" * 72)
    print("R-MEGD-v2 LOW-BUDGET MULTI-SEED PIPELINE")
    print("=" * 72)
    print("Train pool       :", len(train_rows))
    print("Seeds            :", SEEDS)
    print("Budgets          :", BUDGETS)
    print("Boundary threshold:", boundary_threshold)
    print("Global Dev error :", global_error_rate)
    print("Evaluation source:", eval_source_path.name)

    for seed in SEEDS:
        scored_items = build_scored_items(
            train_rows=train_rows,
            loss_rows=loss_rows,
            smoothed_priors=smoothed_priors,
            boundary_threshold=boundary_threshold,
            seed=seed,
        )

        quotas5 = largest_remainder_targets(
            BUDGETS["5pct"]
        )
        anchors5 = choose_signature_anchors(
            items=scored_items,
            quotas=quotas5,
            seed=seed,
            target_n=BUDGETS["5pct"],
        )
        subset5 = fill_subset(
            items=scored_items,
            initial=anchors5,
            quotas=quotas5,
            target_n=BUDGETS["5pct"],
            signature_cap=2,
            query_cap=1,
            seed=seed + 5,
        )

        ids5 = {
            item["sample_id"]
            for item in subset5
        }

        quotas10 = largest_remainder_targets(
            BUDGETS["10pct"]
        )
        subset10 = fill_subset(
            items=scored_items,
            initial=subset5,
            quotas=quotas10,
            target_n=BUDGETS["10pct"],
            signature_cap=4,
            query_cap=2,
            seed=seed + 10,
        )

        ids10 = {
            item["sample_id"]
            for item in subset10
        }

        if not ids5 <= ids10:
            raise RuntimeError(
                f"Nestedness failed for seed {seed}"
            )

        for budget, subset in (
            ("5pct", subset5),
            ("10pct", subset10),
        ):
            cov = coverage(subset)
            expected_n = BUDGETS[budget]
            expected_quotas = largest_remainder_targets(
                expected_n
            )

            if cov["samples"] != expected_n:
                raise RuntimeError(
                    f"{budget} count mismatch"
                )
            if cov["signatures"] != 500:
                raise RuntimeError(
                    f"{budget} lost signature coverage"
                )
            if cov["match"] != expected_quotas:
                raise RuntimeError(
                    f"{budget} match mismatch: "
                    f"{cov['match']} vs {expected_quotas}"
                )

            seed_dir = OUT_ROOT / f"seed_{seed}"
            subset_path = (
                seed_dir
                / f"train_rmegd_{budget}.jsonl"
            )
            write_jsonl(
                subset_path,
                [item["row"] for item in subset],
            )

            manifest_path = (
                seed_dir
                / f"selection_manifest_{budget}.jsonl"
            )
            write_jsonl(
                manifest_path,
                [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "row"
                    }
                    for item in subset
                ],
            )

            run_dir = f"32_rmegd_{budget}_seed_{seed}"
            eval_root = (
                f"33_rmegd_{budget}_seed_{seed}_dev_eval"
            )
            experiment_name = (
                f"tinyllama_qlora_main20k_"
                f"rmegd_v2_{budget}_seed_{seed}"
            )

            train_script = (
                SCRIPT_ROOT
                / f"step48_train_rmegd_{budget}_"
                f"seed_{seed}.py"
            )
            eval_script = (
                SCRIPT_ROOT
                / f"step48_eval_rmegd_{budget}_"
                f"seed_{seed}.py"
            )

            subset_rel = str(
                subset_path.relative_to(HERE)
            )
            train_script.write_text(
                patch_training_script(
                    source=trainer_source,
                    subset_rel=subset_rel,
                    run_dir=run_dir,
                    train_count=expected_n,
                    experiment_name=experiment_name,
                    training_seed=42,
                ),
                encoding="utf-8",
            )
            eval_script.write_text(
                patch_eval_script(
                    source=eval_source,
                    adapter_run_dir=run_dir,
                    eval_root=eval_root,
                ),
                encoding="utf-8",
            )

            master_summary["subsets"][
                f"{budget}_seed_{seed}"
            ] = {
                **cov,
                "subset_file": str(
                    subset_path.relative_to(HERE)
                ),
                "manifest_file": str(
                    manifest_path.relative_to(HERE)
                ),
                "nested_5_in_10": ids5 <= ids10,
            }

            jobs.append({
                "seed": seed,
                "budget": budget,
                "train_script": train_script,
                "eval_script": eval_script,
                "run_dir": HERE / run_dir,
                "eval_dir": HERE / eval_root,
            })

    summary_path = OUT_ROOT / "rmegd_v2_subset_summary.json"
    summary_path.write_text(
        json.dumps(
            master_summary,
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
        run_dir: Path = job["run_dir"]
        eval_dir: Path = job["eval_dir"]

        final_adapter = (
            run_dir
            / "full_run"
            / "final_adapter"
        )
        frozen_dir = (
            run_dir
            / "full_run_frozen"
        )
        eval_summary = (
            eval_dir
            / "test_final_2069"
            / "evaluation_summary.json"
        )

        print()
        print("=" * 72)
        print(
            f"JOB {index}/{len(jobs)} — "
            f"R-MEGD {budget} SEED {seed}"
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

        if not frozen_dir.exists():
            shutil.copytree(
                run_dir / "full_run",
                frozen_dir,
            )
        else:
            print(
                "Frozen model already exists; skipping copy."
            )

        if not eval_summary.exists():
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
            str(eval_summary.relative_to(HERE))
        )

    (OUT_ROOT / "evaluation_result_paths.json").write_text(
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
    print("R-MEGD-v2 MULTI-SEED PIPELINE COMPLETE")
    print("=" * 72)
    print("Blind test used: False")
    print("Subset summary:", summary_path)
    print(
        "Evaluation paths:",
        OUT_ROOT / "evaluation_result_paths.json",
    )


if __name__ == "__main__":
    main()
