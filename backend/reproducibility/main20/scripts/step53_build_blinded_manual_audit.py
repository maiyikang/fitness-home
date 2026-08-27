#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

SEED = 20260818
SAMPLES_PER_GROUP = 10
METHODS = ("M1", "M2", "M3", "M4", "M5")
GROUP_ORDER = (
    "full",
    "weak",
    "partial",
    "near_boundary",
    "cuisine_mismatch",
)

HERE = Path(__file__).resolve().parent
BASELINE_DIR = (
    HERE
    / "41_explanation_baseline_eval"
    / "development_2069_frozen"
)
DEV_FILE = (
    HERE
    / "19_eval_protocol"
    / "development_benchmark_2069.jsonl"
)
OUT_DIR = HERE / "42_explanation_manual_audit"

AUDIT_FILE = OUT_DIR / "manual_audit_250_outputs_blank.csv"
PRIVATE_KEY_FILE = OUT_DIR / "method_key_private.json"
CASE_SUMMARY_FILE = OUT_DIR / "audit_case_summary.json"
GUIDE_FILE = OUT_DIR / "manual_audit_guide.txt"


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


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def checks_of(row: dict[str, Any]) -> dict[str, Any]:
    value = metadata_of(row).get("constraint_checks", {})
    return value if isinstance(value, dict) else {}


def eligible_groups(row: dict[str, Any]) -> set[str]:
    groups: set[str] = set()
    method_metrics = row["metrics"]
    subgroups = method_metrics.get("subgroups", {})

    for group in GROUP_ORDER:
        if bool(subgroups.get(group, False)):
            groups.add(group)

    return groups


def main() -> None:
    required = [DEV_FILE]
    required.extend(
        BASELINE_DIR / f"{method.lower()}_predictions_scored.jsonl"
        for method in METHODS
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    dev_rows = read_jsonl(DEV_FILE)
    dev_by_id = {
        str(row["sample_id"]): row
        for row in dev_rows
    }

    scored_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for method in METHODS:
        rows = read_jsonl(
            BASELINE_DIR
            / f"{method.lower()}_predictions_scored.jsonl"
        )
        scored_by_method[method] = {
            str(row["sample_id"]): row
            for row in rows
        }

    sample_ids = set(dev_by_id)
    for method in METHODS:
        if set(scored_by_method[method]) != sample_ids:
            raise RuntimeError(
                f"{method} sample IDs do not match Development Benchmark."
            )

    # Use M4 only to read subgroup flags; subgroup membership is determined
    # from the source record and is method-independent.
    m4_rows = scored_by_method["M4"]
    pools: dict[str, list[str]] = defaultdict(list)
    for sample_id, row in m4_rows.items():
        for group in eligible_groups(row):
            pools[group].append(sample_id)

    rng = random.Random(SEED)
    selected_ids: set[str] = set()
    selected_cases: list[tuple[str, str]] = []

    # Sample in a fixed order without replacement. Overlapping cases are
    # assigned to the first eligible group in GROUP_ORDER.
    for group in GROUP_ORDER:
        candidates = [
            sample_id
            for sample_id in pools[group]
            if sample_id not in selected_ids
        ]
        rng.shuffle(candidates)
        if len(candidates) < SAMPLES_PER_GROUP:
            raise RuntimeError(
                f"Not enough unique cases for {group}: "
                f"need {SAMPLES_PER_GROUP}, have {len(candidates)}."
            )

        chosen = candidates[:SAMPLES_PER_GROUP]
        for sample_id in chosen:
            selected_ids.add(sample_id)
            selected_cases.append((group, sample_id))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    private_key: dict[str, Any] = {
        "seed": SEED,
        "warning": (
            "Keep this file private until all manual ratings are complete."
        ),
        "cases": {},
    }

    for case_index, (group, sample_id) in enumerate(
        selected_cases,
        start=1,
    ):
        source = dev_by_id[sample_id]
        metadata = metadata_of(source)

        method_order = list(METHODS)
        rng.shuffle(method_order)

        case_id = f"AUDIT{case_index:03d}"
        code_map: dict[str, str] = {}

        for display_index, method in enumerate(method_order, start=1):
            method_code = chr(ord("A") + display_index - 1)
            code_map[method_code] = method
            scored = scored_by_method[method][sample_id]

            audit_rows.append({
                "case_id": case_id,
                "assigned_group": group,
                "display_order": display_index,
                "method_code": method_code,
                "sample_id": sample_id,
                "query": metadata.get("query", ""),
                "restaurant_name": metadata.get("restaurant_name", ""),
                "match_type": metadata.get("match_type", ""),
                "constraints": json.dumps(
                    metadata.get("constraints", {}),
                    ensure_ascii=False,
                ),
                "constraint_checks": json.dumps(
                    checks_of(source),
                    ensure_ascii=False,
                ),
                "structured_evidence_input": source.get("input", ""),
                "candidate_output": scored.get("prediction", ""),
                "factual_correct_0_or_1": "",
                "all_constraints_correct_0_or_1": "",
                "numeric_relations_correct_0_or_1": "",
                "failed_constraints_complete_0_or_1": "",
                "unsupported_claim_0_or_1": "",
                "prompt_leakage_0_or_1": "",
                "naturalness_1_to_5": "",
                "usefulness_1_to_5": "",
                "preference_rank_1_to_5": "",
                "manual_notes": "",
            })

        private_key["cases"][case_id] = {
            "sample_id": sample_id,
            "assigned_group": group,
            "method_code_to_method": code_map,
        }

    fieldnames = list(audit_rows[0].keys())
    with AUDIT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    PRIVATE_KEY_FILE.write_text(
        json.dumps(private_key, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    case_summary = {
        "seed": SEED,
        "case_count": len(selected_cases),
        "output_count": len(audit_rows),
        "methods_per_case": len(METHODS),
        "groups": {
            group: sum(
                selected_group == group
                for selected_group, _ in selected_cases
            )
            for group in GROUP_ORDER
        },
        "method_labels_are_blinded": True,
        "method_order_randomized_per_case": True,
        "development_only": True,
        "blind_test_used": False,
        "important_note": (
            "Subgroups can overlap in the full benchmark, but this audit "
            "assigns each selected case to exactly one group."
        ),
    }
    CASE_SUMMARY_FILE.write_text(
        json.dumps(case_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    guide = """Fitness Home M1-M5 Blinded Manual Audit

Do not open method_key_private.json until every rating is complete.

Each case contains five anonymised candidate outputs (A-E). Evaluate each
candidate only against the supplied structured evidence and constraint checks.

Binary fields:
- factual_correct: restaurant facts and supported claims are correct.
- all_constraints_correct: every required satisfied/failed state is correct.
- numeric_relations_correct: actual values and threshold relationships are correct.
- failed_constraints_complete: every failed constraint is explicitly stated.
- unsupported_claim: 1 if the output introduces any unsupported claim; otherwise 0.
- prompt_leakage: 1 if prompt/instruction/evidence formatting leaks into the answer.

Scale fields:
- naturalness: 1 (very poor) to 5 (excellent).
- usefulness: 1 (not useful) to 5 (very useful).
- preference_rank: rank the five candidates within each case from 1 (best) to 5
  (worst), with no ties.

The Development Benchmark is used here. The reserved Final Blind Test is not used.
"""
    GUIDE_FILE.write_text(guide, encoding="utf-8")

    print("=" * 72)
    print("M1-M5 BLINDED MANUAL AUDIT BUILT")
    print("=" * 72)
    print("Cases             :", len(selected_cases))
    print("Outputs           :", len(audit_rows))
    print("Groups            :", case_summary["groups"])
    print("Methods per case  :", len(METHODS))
    print("Blind test used   : NO")
    print("Audit CSV         :", AUDIT_FILE)
    print("Private method key:", PRIVATE_KEY_FILE)
    print("Guide             :", GUIDE_FILE)


if __name__ == "__main__":
    main()
