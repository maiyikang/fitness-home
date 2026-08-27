from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

SIGNATURE_COUNT = 625
PARAPHRASES_PER_SIGNATURE = 12
CANDIDATES_PER_QUERY = 4
TOP_K = 5
EXPECTED_V4_ACCEPT_RATE = 0.678

HERE = Path(__file__).resolve().parent
WEEK2 = HERE.parent
BACKEND = HERE.parents[2]
WEEK1 = BACKEND / "experiments" / "2026_08_week1"

SIGNATURE_FILE = (
    WEEK1 / "10_ground_truth_2500" / "constraint_signatures_625.jsonl"
)

STEP10_CANDIDATES = [
    WEEK1 / "step10_build_queries_2500_fixed.py",
    WEEK1 / "step10_build_queries_2500.py",
]

OUT_DIR = HERE / "01_query_space"
QUERY_FILE = OUT_DIR / "queries_main20k.jsonl"
RETRIEVAL_FILE = OUT_DIR / "retrieval_results_main20k.jsonl"
SUMMARY_FILE = OUT_DIR / "main20k_query_space_summary.json"

TEMPLATES = [
    "Find a {cuisine} restaurant for {goal} with no more than {cal} calories, at least {protein} g of protein{fiber}.",
    "I want a {cuisine} meal for {goal}. Keep it under {cal} calories with at least {protein} g protein{fiber}.",
    "Recommend a {cuisine} option for {goal}: maximum {cal} kcal and minimum {protein} g protein{fiber}.",
    "For {goal}, look for {cuisine} food below {cal} calories and with at least {protein} g of protein{fiber}.",
    "Search for a {cuisine} restaurant that fits {goal}, capped at {cal} kcal with no less than {protein} g protein{fiber}.",
    "I need a {cuisine} meal for {goal}. The calorie limit is {cal} kcal and the protein minimum is {protein} g{fiber}.",
    "Please find {cuisine} food for {goal} with at most {cal} calories and at least {protein} grams of protein{fiber}.",
    "My target is {goal}. I prefer {cuisine}, no more than {cal} kcal, and at least {protein} g protein{fiber}.",
    "Choose a {cuisine} restaurant for {goal} where the meal stays within {cal} calories and reaches {protein} g protein{fiber}.",
    "For a {goal} meal, find {cuisine} cuisine with a {cal}-kcal maximum and a {protein}-g protein minimum{fiber}.",
    "I am looking for {cuisine} food for {goal}: up to {cal} calories, at least {protein} g protein{fiber}.",
    "Find me a {cuisine} option suitable for my {goal} request, with a maximum of {cal} calories and at least {protein} g protein{fiber}.",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def fiber_clause(value):
    if value is None:
        return ""
    return f", and at least {value} g dietary fibre"


def build_query(template: str, c: dict) -> str:
    return template.format(
        cuisine=c["cuisine"],
        goal=c["goal"],
        cal=c["max_calories"],
        protein=c["min_protein"],
        fiber=fiber_clause(c.get("min_fiber")),
    )


def main():
    if not SIGNATURE_FILE.exists():
        raise FileNotFoundError(SIGNATURE_FILE)

    step10_file = next((p for p in STEP10_CANDIDATES if p.exists()), None)
    if step10_file is None:
        raise FileNotFoundError("Could not find Step-10 query builder.")

    step10 = load_module(step10_file, "fh_step10")
    step2 = step10.load_module(step10.SOURCE_STEP2, "fh_step2")

    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))

    from rag.retriever import retrieve

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    signatures = read_jsonl(SIGNATURE_FILE)
    if len(signatures) != SIGNATURE_COUNT:
        raise RuntimeError(
            f"Expected {SIGNATURE_COUNT} frozen signatures, found {len(signatures)}"
        )

    queries = []
    qn = 0

    for sig in signatures:
        c = sig["constraints"]
        for paraphrase_id, template in enumerate(TEMPLATES, 1):
            qn += 1
            queries.append({
                "query_id": f"M20Q{qn:05d}",
                "constraint_signature_id": sig["constraint_signature_id"],
                "paraphrase_id": paraphrase_id,
                "query": build_query(template, c),
                "constraints": c,
                "ground_truth_size": sig["ground_truth_size"],
                "ground_truth_restaurant_ids": sig[
                    "ground_truth_restaurant_ids"
                ],
            })

    unique_count = len({normalize(x["query"]) for x in queries})
    if unique_count != len(queries):
        raise RuntimeError(
            f"Duplicate queries detected: {unique_count}/{len(queries)} unique"
        )

    write_jsonl(QUERY_FILE, queries)

    completed = {
        row["query_id"]
        for row in read_jsonl(RETRIEVAL_FILE)
    } if RETRIEVAL_FILE.exists() else set()

    raw_target = len(queries) * CANDIDATES_PER_QUERY
    expected_accepted = round(raw_target * EXPECTED_V4_ACCEPT_RATE)

    print("=" * 70)
    print("Fitness Home - Main-20K Query Space")
    print("=" * 70)
    print("Signatures            :", len(signatures))
    print("Paraphrases/signature :", PARAPHRASES_PER_SIGNATURE)
    print("Queries               :", len(queries))
    print("Teacher raw target    :", raw_target)
    print("Expected qualified    :", expected_accepted)
    print("Existing retrievals   :", len(completed))
    print()

    for i, q in enumerate(queries, 1):
        if q["query_id"] in completed:
            continue

        raw = retrieve(q["query"], top_k=TOP_K)
        relevant_ids = set(q["ground_truth_restaurant_ids"])

        evaluated = step10.evaluate_results(
            step2=step2,
            raw_results=raw,
            constraints=q["constraints"],
            relevant_ids=relevant_ids,
        )

        append_jsonl(
            RETRIEVAL_FILE,
            {**q, "retrieval_results": evaluated},
        )

        if i % 100 == 0 or i == len(queries):
            print(f"[{i}/{len(queries)}] retrieval complete")

    results = read_jsonl(RETRIEVAL_FILE)
    if len(results) != len(queries):
        raise RuntimeError(
            f"Retrieval count mismatch: {len(results)}/{len(queries)}"
        )

    best_match = Counter()
    top1 = Counter()
    relevant_top5 = 0

    for row in results:
        retrieved = row.get("retrieval_results", [])

        if retrieved:
            top1[str(retrieved[0].get("restaurant_name", ""))] += 1

        if any(bool(x.get("is_relevant")) for x in retrieved):
            relevant_top5 += 1

        types = [str(x.get("match_type", "partial")) for x in retrieved]
        if "full" in types:
            best_match["full"] += 1
        elif "weak" in types:
            best_match["weak"] += 1
        else:
            best_match["partial"] += 1

    summary = {
        "main_dataset_target": "approximately_20k_qualified_samples",
        "signature_count": len(signatures),
        "paraphrases_per_signature": PARAPHRASES_PER_SIGNATURE,
        "query_count": len(queries),
        "teacher_candidates_per_query": CANDIDATES_PER_QUERY,
        "teacher_raw_target": raw_target,
        "expected_qualified_at_teacher_v4_pilot_rate": expected_accepted,
        "teacher_v4_pilot_acceptance_rate": EXPECTED_V4_ACCEPT_RATE,
        "relevant_in_top5_count": relevant_top5,
        "relevant_in_top5_rate": relevant_top5 / len(results),
        "best_match_distribution": dict(best_match),
        "unique_top1_restaurants": len(top1),
        "most_common_top1": top1.most_common(1)[0] if top1 else None,
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("MAIN-20K QUERY SPACE COMPLETE")
    print("=" * 70)
    print("Signatures         :", len(signatures))
    print("Queries            :", len(queries))
    print("Teacher raw target :", raw_target)
    print("Expected qualified :", expected_accepted)
    print("Relevant in Top-5  :", f"{relevant_top5}/{len(results)}")
    print("Best match         :", dict(best_match))
    print("Unique Top-1       :", len(top1))
    print("Most common Top-1  :", top1.most_common(1)[0] if top1 else None)
    print("Summary            :", SUMMARY_FILE)


if __name__ == "__main__":
    main()
