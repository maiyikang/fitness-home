from __future__ import annotations
import json, random, re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260813
TARGET = 20000

HERE = Path(__file__).resolve().parent
SRC = HERE / "02_teacher_v4_raw30k" / "teacher_v4_filter_v23_accepted.jsonl"
OUT_DIR = HERE / "03_main20k_frozen"
OUT = OUT_DIR / "main20k_frozen.jsonl"
SUMMARY = OUT_DIR / "main20k_frozen_summary.json"

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_output(text):
    text = str(text or "").lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()

def main():
    rows = read_jsonl(SRC)

    # 1) Exact-output deduplication.
    seen = set()
    unique = []
    for r in rows:
        key = norm_output(r.get("output", ""))
        if key and key not in seen:
            seen.add(key)
            unique.append(r)

    if len(unique) < TARGET:
        raise RuntimeError(f"Only {len(unique)} unique samples; need {TARGET}")

    # 2) Deterministic trimming while protecting coverage.
    # Keep at least one sample from every signature, then fill proportionally.
    by_sig = defaultdict(list)
    for r in unique:
        sid = str(r.get("metadata", {}).get("constraint_signature_id", ""))
        by_sig[sid].append(r)

    rng = random.Random(SEED)
    for bucket in by_sig.values():
        rng.shuffle(bucket)

    selected = []
    selected_ids = set()

    # Coverage protection: one sample per signature.
    for sid in sorted(by_sig):
        r = by_sig[sid][0]
        selected.append(r)
        selected_ids.add(r["sample_id"])

    # Fill remaining slots, balancing match type against the unique pool.
    pool_match = Counter(
        str(r.get("metadata", {}).get("match_type", "unknown")) for r in unique
    )
    target_match = {
        k: round(TARGET * v / len(unique))
        for k, v in pool_match.items()
    }
    # Fix rounding to exactly TARGET.
    diff = TARGET - sum(target_match.values())
    if diff:
        largest = pool_match.most_common(1)[0][0]
        target_match[largest] += diff

    current_match = Counter(
        str(r.get("metadata", {}).get("match_type", "unknown")) for r in selected
    )

    remaining = [r for r in unique if r["sample_id"] not in selected_ids]
    rng.shuffle(remaining)

    # First satisfy match-type targets.
    for match in ("full", "weak", "partial"):
        need = max(0, target_match.get(match, 0) - current_match.get(match, 0))
        candidates = [
            r for r in remaining
            if str(r.get("metadata", {}).get("match_type", "")) == match
        ][:need]
        for r in candidates:
            selected.append(r)
            selected_ids.add(r["sample_id"])
        remaining = [r for r in remaining if r["sample_id"] not in selected_ids]

    # Fill any residual slots.
    if len(selected) < TARGET:
        selected.extend(remaining[: TARGET - len(selected)])

    selected = selected[:TARGET]

    if len(selected) != TARGET:
        raise RuntimeError(f"Frozen count mismatch: {len(selected)}")

    frozen_sigs = {
        str(r.get("metadata", {}).get("constraint_signature_id", ""))
        for r in selected
    }
    frozen_queries = {
        str(r.get("metadata", {}).get("query_id", ""))
        for r in selected
    }
    frozen_restaurants = {
        str(r.get("metadata", {}).get("restaurant_name", ""))
        for r in selected
    }
    frozen_match = Counter(
        str(r.get("metadata", {}).get("match_type", "unknown"))
        for r in selected
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT, selected)

    summary = {
        "source_accepted": len(rows),
        "unique_after_exact_output_dedup": len(unique),
        "exact_duplicate_samples_removed": len(rows) - len(unique),
        "frozen_count": len(selected),
        "unique_signatures": len(frozen_sigs - {""}),
        "unique_queries": len(frozen_queries - {""}),
        "unique_restaurants": len(frozen_restaurants - {""}),
        "match_distribution": dict(frozen_match),
        "target_match_distribution": target_match,
        "seed": SEED,
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 70)
    print("MAIN-20K FROZEN")
    print("=" * 70)
    print("Source accepted :", len(rows))
    print("Unique after dedup:", len(unique))
    print("Duplicates removed:", len(rows) - len(unique))
    print("Frozen count    :", len(selected))
    print("Unique signatures:", len(frozen_sigs - {""}))
    print("Unique queries  :", len(frozen_queries - {""}))
    print("Unique restaurants:", len(frozen_restaurants - {""}))
    print("Match           :", dict(frozen_match))
    print("Output          :", OUT)
    print("Summary         :", SUMMARY)

if __name__ == "__main__":
    main()
