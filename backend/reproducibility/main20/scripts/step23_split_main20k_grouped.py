from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SEED = 20260813

HERE = Path(__file__).resolve().parent
SRC = HERE / "03_main20k_frozen" / "main20k_frozen.jsonl"
OUT_DIR = HERE / "04_main20k_split"

TRAIN_FILE = OUT_DIR / "train.jsonl"
VAL_FILE = OUT_DIR / "validation.jsonl"
TEST_FILE = OUT_DIR / "test.jsonl"
SUMMARY_FILE = OUT_DIR / "split_summary.json"
CHECKSUM_FILE = OUT_DIR / "sha256sums.txt"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sig(row):
    return str(row.get("metadata", {}).get("constraint_signature_id", ""))


def match_counts(rows):
    return dict(Counter(
        str(r.get("metadata", {}).get("match_type", "unknown"))
        for r in rows
    ))


def query_count(rows):
    return len({
        str(r.get("metadata", {}).get("query_id", ""))
        for r in rows
        if str(r.get("metadata", {}).get("query_id", ""))
    })


def restaurant_count(rows):
    return len({
        str(r.get("metadata", {}).get("restaurant_name", ""))
        for r in rows
        if str(r.get("metadata", {}).get("restaurant_name", ""))
    })


def main():
    rows = read_jsonl(SRC)
    if len(rows) != 20000:
        raise RuntimeError(f"Expected 20000 frozen samples, got {len(rows)}")

    signatures = sorted({sig(r) for r in rows if sig(r)})
    if len(signatures) != 625:
        raise RuntimeError(f"Expected 625 signatures, got {len(signatures)}")

    rng = random.Random(SEED)
    rng.shuffle(signatures)

    # 625 signatures -> 500 / 62 / 63.
    train_sigs = set(signatures[:500])
    val_sigs = set(signatures[500:562])
    test_sigs = set(signatures[562:])

    train = [r for r in rows if sig(r) in train_sigs]
    val = [r for r in rows if sig(r) in val_sigs]
    test = [r for r in rows if sig(r) in test_sigs]

    if len(train) + len(val) + len(test) != len(rows):
        raise RuntimeError("Split count mismatch")

    if train_sigs & val_sigs or train_sigs & test_sigs or val_sigs & test_sigs:
        raise RuntimeError("Constraint-signature leakage detected")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_FILE, train)
    write_jsonl(VAL_FILE, val)
    write_jsonl(TEST_FILE, test)

    summary = {
        "seed": SEED,
        "group_key": "constraint_signature_id",
        "total_samples": len(rows),
        "total_signatures": len(signatures),
        "train": {
            "samples": len(train),
            "signatures": len(train_sigs),
            "queries": query_count(train),
            "restaurants": restaurant_count(train),
            "match": match_counts(train),
        },
        "validation": {
            "samples": len(val),
            "signatures": len(val_sigs),
            "queries": query_count(val),
            "restaurants": restaurant_count(val),
            "match": match_counts(val),
        },
        "test": {
            "samples": len(test),
            "signatures": len(test_sigs),
            "queries": query_count(test),
            "restaurants": restaurant_count(test),
            "match": match_counts(test),
        },
        "leakage_checks": {
            "train_val_signature_overlap": len(train_sigs & val_sigs),
            "train_test_signature_overlap": len(train_sigs & test_sigs),
            "val_test_signature_overlap": len(val_sigs & test_sigs),
        },
        "important_rule": "The test split is frozen and must not be used for prompt/filter/training decisions.",
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    files = [TRAIN_FILE, VAL_FILE, TEST_FILE, SUMMARY_FILE]
    CHECKSUM_FILE.write_text(
        "\n".join(f"{sha256(p)}  {p.name}" for p in files) + "\n",
        encoding="utf-8",
    )

    print("=" * 70)
    print("MAIN-20K GROUPED SPLIT COMPLETE")
    print("=" * 70)
    print("Train      :", len(train), "samples /", len(train_sigs), "signatures")
    print("Validation :", len(val), "samples /", len(val_sigs), "signatures")
    print("Test       :", len(test), "samples /", len(test_sigs), "signatures")
    print("Train match:", match_counts(train))
    print("Val match  :", match_counts(val))
    print("Test match :", match_counts(test))
    print("Signature overlap train/val :", len(train_sigs & val_sigs))
    print("Signature overlap train/test:", len(train_sigs & test_sigs))
    print("Signature overlap val/test  :", len(val_sigs & test_sigs))
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()
