from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 20260813
BLIND_SIGNATURE_COUNT = 250

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
WEEK1 = BACKEND / "experiments" / "2026_08_week1"

DEV_SOURCE = HERE / "04_main20k_split" / "test.jsonl"
USED_SIGNATURE_FILE = (
    WEEK1 / "10_ground_truth_2500" / "constraint_signatures_625.jsonl"
)

STEP10_CANDIDATES = [
    WEEK1 / "step10_build_queries_2500_fixed.py",
    WEEK1 / "step10_build_queries_2500.py",
]

OUT_DIR = HERE / "19_eval_protocol"
DEV_FILE = OUT_DIR / "development_benchmark_2069.jsonl"
BLIND_SIGNATURE_FILE = OUT_DIR / "reserved_blind_signatures_250.jsonl"
PROTOCOL_FILE = OUT_DIR / "evaluation_protocol.json"
CHECKSUM_FILE = OUT_DIR / "sha256sums.txt"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature_key(constraints: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(constraints.get("cuisine", "")).strip().lower(),
        int(constraints.get("max_calories")),
        int(constraints.get("min_protein")),
        (
            None
            if constraints.get("min_fiber") is None
            else int(constraints.get("min_fiber"))
        ),
        str(constraints.get("goal", "")).strip().lower(),
    )


def main() -> None:
    for path in (DEV_SOURCE, USED_SIGNATURE_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    step10_file = next((p for p in STEP10_CANDIDATES if p.exists()), None)
    if step10_file is None:
        raise FileNotFoundError("Step-10 query builder not found.")

    step10 = load_module(step10_file, "fh_step10_for_blind_reserve")
    step2 = step10.load_module(step10.SOURCE_STEP2, "fh_step2_for_blind_reserve")

    restaurants = step2.load_restaurants()
    pool = step2.build_constraint_pool(restaurants)
    base_specs = step10.collect_all_base_specs(pool)

    goals = list(getattr(step2, "FITNESS_GOALS", step10.DEFAULT_GOALS))
    if not goals:
        goals = list(step10.DEFAULT_GOALS)

    used_rows = read_jsonl(USED_SIGNATURE_FILE)
    used_keys = {
        signature_key(dict(row["constraints"]))
        for row in used_rows
    }

    all_entries: dict[tuple[Any, ...], dict[str, Any]] = {}

    for spec in base_specs:
        base_constraints = dict(spec["constraints"])
        ground_truth = [dict(x) for x in spec["ground_truth"]]
        ground_truth_ids = sorted(step10.ground_truth_ids(ground_truth))

        for goal in goals:
            constraints = {
                **base_constraints,
                "goal": goal,
            }
            key = signature_key(constraints)

            if key not in all_entries:
                all_entries[key] = {
                    "constraints": constraints,
                    "ground_truth_size": len(ground_truth),
                    "ground_truth_restaurant_ids": ground_truth_ids,
                }

    unused = [
        entry
        for key, entry in all_entries.items()
        if key not in used_keys
    ]

    if len(unused) < BLIND_SIGNATURE_COUNT:
        raise RuntimeError(
            f"Only {len(unused)} unused signatures are available; "
            f"need {BLIND_SIGNATURE_COUNT}."
        )

    rng = random.Random(SEED)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for entry in unused:
        constraints = entry["constraints"]
        bucket_key = (
            str(constraints["cuisine"]),
            str(constraints["goal"]),
        )
        buckets[bucket_key].append(entry)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    bucket_keys = sorted(buckets)

    while len(selected) < BLIND_SIGNATURE_COUNT:
        progress = False

        for key in bucket_keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                progress = True

                if len(selected) >= BLIND_SIGNATURE_COUNT:
                    break

        if not progress:
            break

    if len(selected) != BLIND_SIGNATURE_COUNT:
        raise RuntimeError(
            f"Blind reserve count mismatch: {len(selected)}"
        )

    blind_rows = []

    for index, entry in enumerate(selected, 1):
        blind_rows.append({
            "blind_signature_id": f"BSIG{index:04d}",
            **entry,
        })

    selected_keys = {
        signature_key(dict(row["constraints"]))
        for row in blind_rows
    }

    if selected_keys & used_keys:
        raise RuntimeError("Blind signature leakage into Main-20K detected.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEV_SOURCE, DEV_FILE)
    write_jsonl(BLIND_SIGNATURE_FILE, blind_rows)

    cuisine_counts = Counter(
        str(row["constraints"]["cuisine"])
        for row in blind_rows
    )
    goal_counts = Counter(
        str(row["constraints"]["goal"])
        for row in blind_rows
    )

    protocol = {
        "protocol_version": "fitness_home_eval_protocol_v1",
        "seed": SEED,
        "development_benchmark": {
            "file": DEV_FILE.name,
            "samples": len(read_jsonl(DEV_FILE)),
            "status": (
                "Development benchmark only. It has already been used "
                "for model and data-selection decisions and must not be "
                "reported as the untouched final blind test."
            ),
        },
        "final_blind_reserve": {
            "file": BLIND_SIGNATURE_FILE.name,
            "reserved_signatures": len(blind_rows),
            "overlap_with_main20k_signatures": 0,
            "cuisine_distribution": dict(cuisine_counts),
            "goal_distribution": dict(goal_counts),
            "future_raw_target": len(blind_rows) * 4,
            "expected_qualified_range": "approximately 600-800 samples",
        },
        "blind_generation_rules_frozen_before_ctid": [
            "Use only the 250 reserved unseen constraint signatures.",
            "Use new query templates not used by Main-20K training data.",
            "Generate one query per reserved signature.",
            "Select four candidates using structural criteria only; do not use model predictions.",
            "Candidate selection should cover Full, Weak, Partial, near-boundary, cuisine-mismatch, and multi-failure cases where available.",
            "Use the frozen Teacher-v4 prompt and Filter-v2.3 quality gate.",
            "Do not evaluate CTID, random baselines, or tune thresholds on the final blind benchmark before the method is frozen.",
            "Run final method comparison once, then report all outcomes, including negative results.",
        ],
        "future_primary_metrics": [
            "All-Constraint Exact Accuracy",
            "Constraint-State Macro-F1",
            "Numeric Relation Accuracy",
            "Failed-Constraint Recall",
            "Unsupported Goal/Health Claim Rate",
            "Faithfulness",
            "Hallucination Rate",
            "Format Success",
            "ROUGE-L",
        ],
    }

    PROTOCOL_FILE.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    files = [DEV_FILE, BLIND_SIGNATURE_FILE, PROTOCOL_FILE]

    CHECKSUM_FILE.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("DEVELOPMENT / FINAL-BLIND PROTOCOL FROZEN")
    print("=" * 72)
    print("Development samples     :", protocol["development_benchmark"]["samples"])
    print("All candidate signatures:", len(all_entries))
    print("Main-20K used signatures:", len(used_keys))
    print("Unused signatures       :", len(unused))
    print("Blind reserved          :", len(blind_rows))
    print("Blind/Main overlap      :", 0)
    print("Cuisine distribution    :", dict(cuisine_counts))
    print("Goal distribution       :", dict(goal_counts))
    print("Protocol                :", PROTOCOL_FILE)
    print("Checksums               :", CHECKSUM_FILE)


if __name__ == "__main__":
    main()
