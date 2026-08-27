from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROFILE_DIR = HERE / "20_model_error_profile"

PROFILE_FILE = PROFILE_DIR / "base_model_error_profile.jsonl"
OUT_DIR = HERE / "21_hard_dev_profile"

CHALLENGE_FILE = OUT_DIR / "hard_development_challenge.jsonl"
PRIOR_FILE = OUT_DIR / "structural_error_priors.json"
SUMMARY_FILE = OUT_DIR / "hard_dev_summary.json"

HIGH_CONFIDENCE_EVAL_ERRORS = {
    "evaluation_faithfulness_failure",
    "evaluation_hallucination",
    "evaluation_numeric_failure",
    "evaluation_constraint_omission",
    "evaluation_failed_constraint_omission",
    "evaluation_restaurant_missing",
    "evaluation_prompt_leakage",
    "evaluation_format_failure",
}

HIGH_CONFIDENCE_STRICT_ERRORS = {
    "strict_unsupported_numbers",
    "strict_numeric_relation_error",
    "strict_constraint_contradiction",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def high_confidence_errors(row: dict[str, Any]) -> list[str]:
    labels = set(row.get("error_labels", []))
    keep = sorted(
        (labels & HIGH_CONFIDENCE_EVAL_ERRORS)
        | (labels & HIGH_CONFIDENCE_STRICT_ERRORS)
    )
    return keep


def structural_key(row: dict[str, Any]) -> str:
    f = row["structural_features"]
    failed = int(f.get("failed_constraint_count", 0))

    if failed == 0:
        failed_bucket = "fail0"
    elif failed == 1:
        failed_bucket = "fail1"
    else:
        failed_bucket = "fail2plus"

    return "|".join([
        str(f.get("match_type", "unknown")),
        failed_bucket,
        "boundary" if f.get("near_boundary") else "nonboundary",
        "cuisine_mismatch" if f.get("cuisine_mismatch") else "cuisine_match",
    ])


def main() -> None:
    if not PROFILE_FILE.exists():
        raise FileNotFoundError(PROFILE_FILE)

    rows = read_jsonl(PROFILE_FILE)
    if len(rows) != 2069:
        raise RuntimeError(f"Expected 2069 profile rows, got {len(rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    challenge = []
    high_conf_counter = Counter()
    bin_total = Counter()
    bin_error = Counter()
    match_error = Counter()
    match_total = Counter()

    for row in rows:
        errors = high_confidence_errors(row)
        features = row["structural_features"]
        key = structural_key(row)
        match = str(features.get("match_type", "unknown"))

        bin_total[key] += 1
        match_total[match] += 1

        has_high_conf_error = bool(errors)

        if has_high_conf_error:
            bin_error[key] += 1
            match_error[match] += 1
            high_conf_counter.update(errors)

        is_structural_anchor = bool(
            features.get("near_boundary")
            or features.get("multiple_failures")
            or features.get("cuisine_mismatch")
        )

        if has_high_conf_error or is_structural_anchor:
            out = dict(row)
            out["high_confidence_error_labels"] = errors
            out["has_high_confidence_error"] = has_high_conf_error
            out["structural_bin"] = key
            challenge.append(out)

    priors = {}
    for key in sorted(bin_total):
        total = bin_total[key]
        errors = bin_error[key]
        priors[key] = {
            "samples": total,
            "high_confidence_errors": errors,
            "error_rate": errors / total if total else 0.0,
        }

    write_jsonl(CHALLENGE_FILE, challenge)

    PRIOR_FILE.write_text(
        json.dumps(
            {
                "definition": (
                    "High-confidence errors use frozen evaluation metrics plus "
                    "strict numeric/constraint contradiction checks. Broad "
                    "goal-health keyword flags are excluded from the primary "
                    "error target to avoid over-counting."
                ),
                "structural_priors": priors,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "development_samples": len(rows),
        "challenge_samples": len(challenge),
        "high_confidence_error_samples": sum(
            1 for r in challenge if r["has_high_confidence_error"]
        ),
        "structural_anchor_only_samples": sum(
            1 for r in challenge if not r["has_high_confidence_error"]
        ),
        "high_confidence_error_type_counts": dict(
            high_conf_counter.most_common()
        ),
        "match_error_rates": {
            match: {
                "samples": match_total[match],
                "high_confidence_error_samples": match_error[match],
                "error_rate": (
                    match_error[match] / match_total[match]
                    if match_total[match]
                    else 0.0
                ),
            }
            for match in sorted(match_total)
        },
        "top_structural_bins_by_error_rate": sorted(
            [
                {
                    "bin": key,
                    **value,
                }
                for key, value in priors.items()
                if value["samples"] >= 10
            ],
            key=lambda x: (x["error_rate"], x["samples"]),
            reverse=True,
        )[:15],
        "important_note": (
            "This is a development-only challenge profile. It does not access "
            "the reserved final blind signatures."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("HARD DEVELOPMENT PROFILE FROZEN")
    print("=" * 72)
    print("Development samples          :", len(rows))
    print("Challenge samples            :", len(challenge))
    print(
        "High-confidence error samples:",
        summary["high_confidence_error_samples"],
    )
    print(
        "Structural anchor-only samples:",
        summary["structural_anchor_only_samples"],
    )
    print("High-confidence error types:")
    for key, value in high_conf_counter.most_common():
        print(f"  {key:<45} {value}")

    print("Match error rates:")
    for match, value in summary["match_error_rates"].items():
        print(
            f"  {match:<8} "
            f"{value['high_confidence_error_samples']}/"
            f"{value['samples']} = {value['error_rate']:.2%}"
        )

    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
