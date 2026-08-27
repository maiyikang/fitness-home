from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WEEK2 = HERE.parent

DEV_FILE = (
    HERE
    / "19_eval_protocol"
    / "development_benchmark_2069.jsonl"
)
BASE_PREDICTIONS_FILE = (
    HERE
    / "06_main20k_eval"
    / "test_final_2069"
    / "base_predictions_scored.jsonl"
)
FILTER_FILE = (
    WEEK2
    / "01_filter_v2"
    / "step14_filter_v2_3_calibration.py"
)

OUT_DIR = HERE / "20_model_error_profile"
ALL_PROFILE_FILE = OUT_DIR / "base_model_error_profile.jsonl"
ERROR_TARGET_FILE = OUT_DIR / "base_model_error_targets.jsonl"
CORRECT_ANCHOR_FILE = OUT_DIR / "correct_challenge_anchors.jsonl"
SUMMARY_FILE = OUT_DIR / "base_model_error_profile_summary.json"
ERROR_COUNTS_FILE = OUT_DIR / "error_type_counts.csv"

CAL_RE = re.compile(
    r"- Average calories:\s*(\d+(?:\.\d+)?)\s*kcal",
    re.I,
)
PRO_RE = re.compile(
    r"- Average protein:\s*(\d+(?:\.\d+)?)\s*g",
    re.I,
)
FIB_RE = re.compile(
    r"- Average fibre:\s*(\d+(?:\.\d+)?)\s*g",
    re.I,
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")

    module = importlib.util.module_from_spec(spec)
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


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def constraints_of(row: dict[str, Any]) -> dict[str, Any]:
    value = metadata_of(row).get("constraints", {})
    return value if isinstance(value, dict) else {}


def checks_of(row: dict[str, Any]) -> dict[str, bool]:
    value = metadata_of(row).get("constraint_checks", {})
    return value if isinstance(value, dict) else {}


def parse_actuals(text: str) -> dict[str, float | None]:
    def get(pattern: re.Pattern[str]) -> float | None:
        match = pattern.search(text)
        return float(match.group(1)) if match else None

    return {
        "calories": get(CAL_RE),
        "protein": get(PRO_RE),
        "fiber": get(FIB_RE),
    }


def relative_margin(
    actual: float | None,
    target: Any,
) -> float | None:
    if actual is None or target is None:
        return None

    target_value = float(target)

    return abs(actual - target_value) / max(
        abs(target_value),
        1.0,
    )


def structural_features(
    source: dict[str, Any],
) -> dict[str, Any]:
    constraints = constraints_of(source)
    checks = checks_of(source)
    actual = parse_actuals(
        str(source.get("input", ""))
    )

    margins = {
        "calories": relative_margin(
            actual.get("calories"),
            constraints.get("max_calories"),
        ),
        "protein": relative_margin(
            actual.get("protein"),
            constraints.get("min_protein"),
        ),
        "fiber": relative_margin(
            actual.get("fiber"),
            constraints.get("min_fiber"),
        ),
    }

    available_margins = [
        value
        for value in margins.values()
        if value is not None
    ]

    failed_constraints = sorted(
        name
        for name, passed in checks.items()
        if passed is False
    )

    state_pattern = "|".join(
        f"{name}={'T' if passed else 'F'}"
        for name, passed in sorted(checks.items())
    )

    minimum_margin = (
        min(available_margins)
        if available_margins
        else None
    )

    return {
        "match_type": metadata_of(source).get(
            "match_type"
        ),
        "cuisine": constraints.get("cuisine"),
        "goal": constraints.get("goal"),
        "failed_constraints": failed_constraints,
        "failed_constraint_count": len(
            failed_constraints
        ),
        "constraint_state_pattern": state_pattern,
        "numeric_relative_margins": margins,
        "minimum_relative_margin": minimum_margin,
        "near_boundary": bool(
            minimum_margin is not None
            and minimum_margin <= 0.10
        ),
        "exact_numeric_boundary": any(
            value is not None and value == 0.0
            for value in margins.values()
        ),
        "cuisine_mismatch": (
            checks.get("cuisine") is False
        ),
        "multiple_failures": (
            len(failed_constraints) >= 2
        ),
    }


def normalise_reason(reason: str) -> str:
    return str(reason).split(":", 1)[0]


def derive_error_labels(
    metrics: dict[str, Any],
    strict_reasons: list[str],
) -> list[str]:
    labels: list[str] = []

    if metrics.get("faithfulness_pass") is False:
        labels.append("evaluation_faithfulness_failure")

    if metrics.get("hallucination") is True:
        labels.append("evaluation_hallucination")

    if metrics.get("numeric_faithful") is False:
        labels.append("evaluation_numeric_failure")

    if metrics.get("prompt_leakage") is True:
        labels.append("evaluation_prompt_leakage")

    if metrics.get("restaurant_mentioned") is False:
        labels.append("evaluation_restaurant_missing")

    if metrics.get("format_success") is False:
        labels.append("evaluation_format_failure")

    if float(
        metrics.get("constraint_coverage_rate", 1.0)
    ) < 0.999999:
        labels.append("evaluation_constraint_omission")

    if float(
        metrics.get(
            "failed_constraint_coverage_rate",
            1.0,
        )
    ) < 0.999999:
        labels.append(
            "evaluation_failed_constraint_omission"
        )

    labels.extend(
        f"strict_{normalise_reason(reason)}"
        for reason in strict_reasons
    )

    return list(dict.fromkeys(labels))


def severity_score(
    labels: list[str],
    features: dict[str, Any],
) -> float:
    weights = {
        "strict_unsupported_numbers": 4.0,
        "strict_numeric_relation_error": 4.0,
        "strict_constraint_contradiction": 4.0,
        "evaluation_numeric_failure": 4.0,
        "strict_missing_failed_constraint": 3.5,
        "evaluation_failed_constraint_omission": 3.5,
        "strict_unsupported_goal_or_health_claim": 3.0,
        "evaluation_hallucination": 3.0,
        "evaluation_faithfulness_failure": 3.0,
        "strict_restaurant_not_mentioned": 2.5,
        "evaluation_restaurant_missing": 2.5,
        "evaluation_constraint_omission": 2.0,
        "strict_prompt_leakage": 2.0,
        "evaluation_prompt_leakage": 2.0,
        "evaluation_format_failure": 1.5,
    }

    score = sum(
        weights.get(label, 1.0)
        for label in labels
    )

    if features["near_boundary"]:
        score += 0.5

    if features["multiple_failures"]:
        score += 0.5

    if features["cuisine_mismatch"]:
        score += 0.5

    return round(score, 4)


def main() -> None:
    for path in (
        DEV_FILE,
        BASE_PREDICTIONS_FILE,
        FILTER_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    filter_v23 = load_module(
        FILTER_FILE,
        "fitness_home_filter_v23_profile",
    )

    dev_rows = read_jsonl(DEV_FILE)
    prediction_rows = read_jsonl(
        BASE_PREDICTIONS_FILE
    )

    if len(dev_rows) != 2069:
        raise RuntimeError(
            f"Expected 2069 development rows, "
            f"found {len(dev_rows)}"
        )

    source_by_id = {
        str(row["sample_id"]): row
        for row in dev_rows
    }

    prediction_by_id = {
        str(row["sample_id"]): row
        for row in prediction_rows
    }

    if set(source_by_id) != set(prediction_by_id):
        raise RuntimeError(
            "Development/prediction sample IDs do not match."
        )

    profiles: list[dict[str, Any]] = []
    error_targets: list[dict[str, Any]] = []
    correct_anchors: list[dict[str, Any]] = []

    error_counter: Counter[str] = Counter()
    match_counter: Counter[str] = Counter()
    error_match_counter: Counter[str] = Counter()
    state_counter: Counter[str] = Counter()
    error_state_counter: Counter[str] = Counter()
    cuisine_counter: Counter[str] = Counter()
    goal_counter: Counter[str] = Counter()

    for sample_id in sorted(source_by_id):
        source = source_by_id[sample_id]
        prediction_row = prediction_by_id[sample_id]
        prediction = str(
            prediction_row.get("prediction", "")
        )
        metrics = dict(
            prediction_row.get("metrics", {})
        )

        strict_record = json.loads(
            json.dumps(source, ensure_ascii=False)
        )
        strict_record["output"] = prediction

        strict_reasons = list(
            filter_v23.filter_reasons(strict_record)
        )

        features = structural_features(source)
        labels = derive_error_labels(
            metrics,
            strict_reasons,
        )
        severity = severity_score(
            labels,
            features,
        )

        profile = {
            "sample_id": sample_id,
            "query_id": metadata_of(source).get(
                "query_id"
            ),
            "constraint_signature_id": (
                metadata_of(source).get(
                    "constraint_signature_id"
                )
            ),
            "restaurant_name": metadata_of(
                source
            ).get("restaurant_name"),
            "query": metadata_of(source).get(
                "query"
            ),
            "constraints": constraints_of(source),
            "constraint_checks": checks_of(source),
            "reference": source.get("output"),
            "base_prediction": prediction,
            "evaluation_metrics": metrics,
            "strict_filter_reasons": strict_reasons,
            "error_labels": labels,
            "error_severity_score": severity,
            "structural_features": features,
        }

        profiles.append(profile)

        match = str(
            features.get("match_type", "unknown")
        )
        state = str(
            features.get(
                "constraint_state_pattern",
                "",
            )
        )
        cuisine = str(
            features.get("cuisine", "")
        )
        goal = str(
            features.get("goal", "")
        )

        match_counter[match] += 1
        state_counter[state] += 1
        cuisine_counter[cuisine] += 1
        goal_counter[goal] += 1

        for label in labels:
            error_counter[label] += 1

        if labels:
            error_targets.append(profile)
            error_match_counter[match] += 1
            error_state_counter[state] += 1
        elif (
            features["near_boundary"]
            or features["multiple_failures"]
            or features["cuisine_mismatch"]
        ):
            correct_anchors.append(profile)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        ALL_PROFILE_FILE,
        profiles,
    )
    write_jsonl(
        ERROR_TARGET_FILE,
        error_targets,
    )
    write_jsonl(
        CORRECT_ANCHOR_FILE,
        correct_anchors,
    )

    with ERROR_COUNTS_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["error_type", "count"])

        for error_type, count in (
            error_counter.most_common()
        ):
            writer.writerow([error_type, count])

    summary = {
        "profile_version": (
            "base_tinyllama_development_error_profile_v1"
        ),
        "development_only": True,
        "development_samples": len(profiles),
        "samples_with_any_error": len(
            error_targets
        ),
        "samples_without_detected_error": (
            len(profiles) - len(error_targets)
        ),
        "correct_challenge_anchors": len(
            correct_anchors
        ),
        "error_sample_rate": (
            len(error_targets) / len(profiles)
        ),
        "error_type_counts": dict(
            error_counter.most_common()
        ),
        "all_match_distribution": dict(
            match_counter
        ),
        "error_match_distribution": dict(
            error_match_counter
        ),
        "all_constraint_state_distribution": dict(
            state_counter.most_common()
        ),
        "error_constraint_state_distribution": dict(
            error_state_counter.most_common()
        ),
        "cuisine_distribution": dict(
            cuisine_counter
        ),
        "goal_distribution": dict(
            goal_counter
        ),
        "important_note": (
            "This profile uses only the frozen development "
            "benchmark. Reserved final-blind signatures and "
            "future blind samples are not accessed."
        ),
        "files": {
            "all_profile": ALL_PROFILE_FILE.name,
            "error_targets": ERROR_TARGET_FILE.name,
            "correct_challenge_anchors": (
                CORRECT_ANCHOR_FILE.name
            ),
            "error_counts": ERROR_COUNTS_FILE.name,
        },
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("BASE MODEL DEVELOPMENT ERROR PROFILE")
    print("=" * 72)
    print("Development samples :", len(profiles))
    print("Samples with errors :", len(error_targets))
    print(
        "Error sample rate   :",
        f"{len(error_targets) / len(profiles):.2%}",
    )
    print(
        "Correct hard anchors:",
        len(correct_anchors),
    )
    print(
        "Error match         :",
        dict(error_match_counter),
    )
    print("Top error types:")

    for error_type, count in (
        error_counter.most_common(15)
    ):
        print(
            f"  {error_type:<45} {count}"
        )

    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
