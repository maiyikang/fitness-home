from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260813
TARGETS = {"25pct": 3996, "50pct": 7992}

HERE = Path(__file__).resolve().parent
TRAIN_FILE = HERE / "04_main20k_split" / "train.jsonl"
OUT_DIR = HERE / "14_matched_random_baseline"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def md(row):
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def largest_remainder_targets(full_counts, target_n):
    total = sum(full_counts.values())
    raw = {k: target_n * v / total for k, v in full_counts.items()}
    targets = {k: math.floor(v) for k, v in raw.items()}
    remaining = target_n - sum(targets.values())
    order = sorted(raw, key=lambda k: (raw[k] - targets[k], k), reverse=True)
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def coverage(rows):
    return {
        "signatures": len({
            str(md(r).get("constraint_signature_id", ""))
            for r in rows
        } - {""}),
        "queries": len({
            str(md(r).get("query_id", ""))
            for r in rows
        } - {""}),
        "restaurants": len({
            str(md(r).get("restaurant_name", ""))
            for r in rows
        } - {""}),
        "match": dict(Counter(
            str(md(r).get("match_type", "unknown"))
            for r in rows
        )),
    }


def build_25(rows, by_sig, target_match, rng):
    selected, selected_ids = [], set()
    current = Counter()

    sigs = sorted(by_sig)
    rng.shuffle(sigs)

    for sid in sigs:
        candidates = [
            r for r in by_sig[sid]
            if current[str(md(r).get("match_type", ""))]
            < target_match[str(md(r).get("match_type", ""))]
        ]
        if not candidates:
            candidates = list(by_sig[sid])

        chosen = rng.choice(candidates)
        selected.append(chosen)
        selected_ids.add(str(chosen["sample_id"]))
        current[str(md(chosen).get("match_type", ""))] += 1

    for match in ("full", "weak", "partial"):
        need = target_match[match] - current[match]
        pool = [
            r for r in rows
            if str(r["sample_id"]) not in selected_ids
            and str(md(r).get("match_type", "")) == match
        ]
        for r in rng.sample(pool, need):
            selected.append(r)
            selected_ids.add(str(r["sample_id"]))
            current[match] += 1

    return selected


def extend_nested(rows, selected, target_match, rng):
    selected = list(selected)
    selected_ids = {str(r["sample_id"]) for r in selected}
    current = Counter(str(md(r).get("match_type", "")) for r in selected)

    for match in ("full", "weak", "partial"):
        need = target_match[match] - current[match]
        pool = [
            r for r in rows
            if str(r["sample_id"]) not in selected_ids
            and str(md(r).get("match_type", "")) == match
        ]
        for r in rng.sample(pool, need):
            selected.append(r)
            selected_ids.add(str(r["sample_id"]))
            current[match] += 1

    return selected


def main():
    rows = read_jsonl(TRAIN_FILE)
    if len(rows) != 15983:
        raise RuntimeError(f"Expected 15983 train samples, got {len(rows)}")

    by_sig = defaultdict(list)
    full_match = Counter()

    for row in rows:
        sid = str(md(row).get("constraint_signature_id", ""))
        if not sid:
            raise RuntimeError("Missing constraint_signature_id")
        by_sig[sid].append(row)
        full_match[str(md(row).get("match_type", ""))] += 1

    if len(by_sig) != 500:
        raise RuntimeError(f"Expected 500 signatures, got {len(by_sig)}")

    rng = random.Random(SEED)

    targets25 = largest_remainder_targets(full_match, TARGETS["25pct"])
    subset25 = build_25(rows, by_sig, targets25, rng)

    targets50 = largest_remainder_targets(full_match, TARGETS["50pct"])
    subset50 = extend_nested(rows, subset25, targets50, rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file25 = OUT_DIR / "train_matched_random_25pct.jsonl"
    file50 = OUT_DIR / "train_matched_random_50pct.jsonl"
    write_jsonl(file25, subset25)
    write_jsonl(file50, subset50)

    ids25 = {str(r["sample_id"]) for r in subset25}
    ids50 = {str(r["sample_id"]) for r in subset50}

    summary = {
        "method": "Matched Random Sampling Baseline",
        "seed": SEED,
        "purpose": (
            "Matched to B-CAEGD on sample count, all-500 signature coverage, "
            "Full/Weak/Partial quotas, and nested subset design; "
            "sample choice itself is random."
        ),
        "source_train_samples": len(rows),
        "source_match_distribution": dict(full_match),
        "25pct": {"samples": len(subset25), **coverage(subset25), "file": file25.name},
        "50pct": {"samples": len(subset50), **coverage(subset50), "file": file50.name},
        "nested_checks": {"25_in_50": ids25 <= ids50},
    }

    summary_file = OUT_DIR / "matched_random_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("MATCHED RANDOM BASELINE COMPLETE")
    print("=" * 72)
    print("100% match:", dict(full_match))
    print("50pct:", len(subset50), "|", coverage(subset50))
    print("25pct:", len(subset25), "|", coverage(subset25))
    print("Nested checks:", summary["nested_checks"])
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
