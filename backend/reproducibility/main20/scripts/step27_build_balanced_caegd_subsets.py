from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260813
RATIOS = (0.25, 0.50, 0.75)

HERE = Path(__file__).resolve().parent
TRAIN_FILE = HERE / "04_main20k_split" / "train.jsonl"
OUT_DIR = HERE / "07_caegd_distillation_balanced"

CAL_RE = re.compile(r"- Average calories:\s*(\d+(?:\.\d+)?)\s*kcal", re.I)
PRO_RE = re.compile(r"- Average protein:\s*(\d+(?:\.\d+)?)\s*g", re.I)
FIB_RE = re.compile(r"- Average fibre:\s*(\d+(?:\.\d+)?)\s*g", re.I)


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def md(row):
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def actuals(text):
    def get(rx):
        m = rx.search(text)
        return float(m.group(1)) if m else None

    return {
        "calories": get(CAL_RE),
        "protein": get(PRO_RE),
        "fiber": get(FIB_RE),
    }


def boundary_score(row):
    c = md(row).get("constraints", {}) or {}
    a = actuals(str(row.get("input", "")))

    pairs = [
        (a.get("calories"), c.get("max_calories")),
        (a.get("protein"), c.get("min_protein")),
    ]
    if c.get("min_fiber") is not None:
        pairs.append((a.get("fiber"), c.get("min_fiber")))

    vals = []
    for actual, target in pairs:
        if actual is None or target is None:
            continue
        target = float(target)
        margin = abs(float(actual) - target) / max(abs(target), 1.0)
        vals.append(1.0 / (1.0 + 5.0 * margin))

    return sum(vals) / len(vals) if vals else 0.0


def failed_count(row):
    checks = md(row).get("constraint_checks", {}) or {}
    return sum(value is False for value in checks.values())


def stable_tie(sample_id):
    h = hashlib.sha256(f"{SEED}:{sample_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def utility(row, restaurant_freq, query_freq):
    m = md(row)
    restaurant = str(m.get("restaurant_name", ""))
    query_id = str(m.get("query_id", ""))

    rare_restaurant = 1.0 / math.sqrt(max(restaurant_freq[restaurant], 1))
    rare_query = 1.0 / math.sqrt(max(query_freq[query_id], 1))

    return (
        1.80 * boundary_score(row)
        + 0.45 * failed_count(row)
        + 1.00 * rare_restaurant
        + 0.45 * rare_query
        + 0.01 * stable_tie(str(row.get("sample_id", "")))
    )


def largest_remainder_targets(full_counts, target_n):
    total = sum(full_counts.values())
    raw = {k: target_n * v / total for k, v in full_counts.items()}
    targets = {k: math.floor(v) for k, v in raw.items()}
    remaining = target_n - sum(targets.values())

    order = sorted(
        raw,
        key=lambda k: (raw[k] - targets[k], k),
        reverse=True,
    )
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


def main():
    rows = read_jsonl(TRAIN_FILE)
    if len(rows) != 15983:
        raise RuntimeError(f"Expected 15983 train samples, got {len(rows)}")

    by_sig = defaultdict(list)
    restaurant_freq = Counter()
    query_freq = Counter()
    full_match = Counter()

    for row in rows:
        m = md(row)
        sid = str(m.get("constraint_signature_id", ""))
        match = str(m.get("match_type", ""))

        if not sid:
            raise RuntimeError("Missing constraint_signature_id")

        by_sig[sid].append(row)
        restaurant_freq[str(m.get("restaurant_name", ""))] += 1
        query_freq[str(m.get("query_id", ""))] += 1
        full_match[match] += 1

    if len(by_sig) != 500:
        raise RuntimeError(f"Expected 500 signatures, got {len(by_sig)}")

    score = {
        str(r["sample_id"]): utility(r, restaurant_freq, query_freq)
        for r in rows
    }

    selected_ids = set()
    selected = []
    rng = random.Random(SEED)

    summary = {
        "method": (
            "Balanced Constraint-Aware Evidence-Grounded "
            "Data Distillation (B-CAEGD)"
        ),
        "seed": SEED,
        "source_train_samples": len(rows),
        "source_train_signatures": len(by_sig),
        "source_match_distribution": dict(full_match),
        "design": {
            "quality_gate": (
                "Teacher-v4 + frozen Filter-v2.3 + exact dedup"
            ),
            "coverage": "all 500 train constraint signatures preserved",
            "distribution": (
                "Full/Weak/Partial proportions preserved using "
                "largest-remainder quotas"
            ),
            "hard_examples": (
                "near-boundary numeric cases and failed constraints "
                "receive higher utility"
            ),
            "diversity": (
                "rare restaurants and rare query paraphrases receive "
                "higher utility"
            ),
            "nested": "25% subset of 50%; 50% subset of 75%",
        },
        "subsets": {},
    }

    for ratio in RATIOS:
        target_n = round(len(rows) * ratio)
        target_match = largest_remainder_targets(full_match, target_n)
        current_match = Counter(
            str(md(r).get("match_type", ""))
            for r in selected
        )

        # Stage 1 (25% only): guarantee one sample per signature.
        if ratio == 0.25:
            sigs = sorted(by_sig)
            rng.shuffle(sigs)

            for sid in sigs:
                candidates = [
                    r for r in by_sig[sid]
                    if str(r["sample_id"]) not in selected_ids
                    and current_match[
                        str(md(r).get("match_type", ""))
                    ] < target_match[
                        str(md(r).get("match_type", ""))
                    ]
                ]

                if not candidates:
                    candidates = [
                        r for r in by_sig[sid]
                        if str(r["sample_id"]) not in selected_ids
                    ]

                if not candidates:
                    raise RuntimeError(
                        f"No candidate left for signature {sid}"
                    )

                chosen = max(
                    candidates,
                    key=lambda r: score[str(r["sample_id"])],
                )

                selected.append(chosen)
                selected_ids.add(str(chosen["sample_id"]))
                current_match[
                    str(md(chosen).get("match_type", ""))
                ] += 1

        # Stage 2: fill exact Full/Weak/Partial quotas.
        for match in ("full", "weak", "partial"):
            need = target_match[match] - current_match[match]
            if need < 0:
                raise RuntimeError(
                    f"Quota exceeded for {match}: "
                    f"{current_match[match]} > {target_match[match]}"
                )

            candidates = [
                r for r in rows
                if str(r["sample_id"]) not in selected_ids
                and str(md(r).get("match_type", "")) == match
            ]
            candidates.sort(
                key=lambda r: score[str(r["sample_id"])],
                reverse=True,
            )

            if len(candidates) < need:
                raise RuntimeError(
                    f"Not enough {match} candidates: "
                    f"need {need}, have {len(candidates)}"
                )

            for chosen in candidates[:need]:
                selected.append(chosen)
                selected_ids.add(str(chosen["sample_id"]))
                current_match[match] += 1

        if len(selected) != target_n:
            raise RuntimeError(
                f"Stage {ratio:.2f}: expected {target_n}, "
                f"got {len(selected)}"
            )

        actual_match = Counter(
            str(md(r).get("match_type", ""))
            for r in selected
        )

        if dict(actual_match) != dict(target_match):
            raise RuntimeError(
                f"Match quota mismatch at {ratio:.2f}: "
                f"{dict(actual_match)} vs {dict(target_match)}"
            )

        label = f"{int(ratio * 100)}pct"
        path = OUT_DIR / f"train_bcaegd_{label}.jsonl"

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, selected)

        summary["subsets"][label] = {
            "samples": len(selected),
            "target_match": dict(target_match),
            **coverage(selected),
            "file": path.name,
        }

    ids25 = {
        str(r["sample_id"])
        for r in read_jsonl(OUT_DIR / "train_bcaegd_25pct.jsonl")
    }
    ids50 = {
        str(r["sample_id"])
        for r in read_jsonl(OUT_DIR / "train_bcaegd_50pct.jsonl")
    }
    ids75 = {
        str(r["sample_id"])
        for r in read_jsonl(OUT_DIR / "train_bcaegd_75pct.jsonl")
    }
    full_ids = {str(r["sample_id"]) for r in rows}

    summary["nested_checks"] = {
        "25_in_50": ids25 <= ids50,
        "50_in_75": ids50 <= ids75,
        "75_in_100": ids75 <= full_ids,
    }

    summary_file = OUT_DIR / "bcaegd_distillation_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("B-CAEGD DISTILLATION COMPLETE")
    print("=" * 72)
    print("100% match:", dict(full_match))

    for label in ("75pct", "50pct", "25pct"):
        s = summary["subsets"][label]
        print(
            f"{label:<5}: {s['samples']} samples | "
            f"{s['signatures']} signatures | "
            f"{s['queries']} queries | "
            f"{s['restaurants']} restaurants | "
            f"{s['match']}"
        )

    print("Nested checks:", summary["nested_checks"])
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
