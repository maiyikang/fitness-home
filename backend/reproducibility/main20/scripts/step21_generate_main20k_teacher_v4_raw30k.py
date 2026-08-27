from __future__ import annotations

import gc
import importlib.util
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

SEED = 20260812
BATCH_SIZE = 10
CANDIDATES_PER_QUERY = 4

HERE = Path(__file__).resolve().parent
WEEK2 = HERE.parent
BACKEND = HERE.parents[2]
WEEK1 = BACKEND / "experiments" / "2026_08_week1"

INPUT_FILE = HERE / "01_query_space" / "retrieval_results_main20k.jsonl"
STEP11_FILE = WEEK1 / "step11_generate_teacher_10k_batch10.py"
FILTER_FILE = WEEK2 / "01_filter_v2" / "step14_filter_v2_3_calibration.py"

OUT_DIR = HERE / "02_teacher_v4_raw30k"
RAW_FILE = OUT_DIR / "teacher_v4_raw_30k.jsonl"
ACC_FILE = OUT_DIR / "teacher_v4_filter_v23_accepted.jsonl"
REJ_FILE = OUT_DIR / "teacher_v4_filter_v23_rejected.jsonl"
SUMMARY_FILE = OUT_DIR / "teacher_v4_raw30k_summary.json"

V4_SYSTEM_PROMPT = """
You are the Teacher Model for Fitness Home.
The retrieval system has already selected the restaurant.
Explain only the supplied evidence and precomputed constraint evaluation.

Strict rules:
1. Never invent or alter restaurant facts, cuisine, category, tags, calories,
   protein, fibre, menu items, ingredients, prices, or labels.
2. A fitness goal (fat loss, muscle gain, weight maintenance, post-workout
   recovery) is only a user-request label. Do not claim the restaurant is
   suitable, useful, beneficial, supportive, aligned, sufficient, or effective
   for that goal unless explicitly stated as evidence.
3. Do not infer muscle repair, recovery effects, digestive health, general
   health, daily nutritional needs, balanced nutrition, or other health effects.
4. Follow constraint_checks exactly. If cuisine=false, explicitly state the
   cuisine requirement is not satisfied; never call it the requested cuisine.
   Do not reinterpret category or cuisine tags.
5. Compare numbers literally: actual < maximum means below/within the limit;
   actual > maximum means exceeds the limit; actual = minimum means meets the
   minimum, not exceeds it; actual < minimum means below the minimum.
6. If a nutrient is not requested, do not judge it as low, high, useful,
   healthy, beneficial, sufficient, or insufficient.
7. For weak/partial matches, never state that all requirements are met and
   clearly state every failed constraint.
8. Do not convert a meal-level calorie limit into a daily requirement.
9. Return one concise paragraph only.
""".strip()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_tasks(teacher, query_rows):
    rng = random.Random(SEED)
    tasks = []

    for q in query_rows:
        results = teacher.select_results_for_query(
            query_record=q,
            count=CANDIDATES_PER_QUERY,
            rng=rng,
        )
        if len(results) != CANDIDATES_PER_QUERY:
            raise RuntimeError(
                f"{q.get('query_id')} returned {len(results)} candidates"
            )

        for i, result in enumerate(results, 1):
            rank = int(result.get("rank") or i)
            tasks.append({
                "sample_id": f"M20_{q['query_id']}_R{rank:02d}_S{i:02d}",
                "query_record": q,
                "selected_result": dict(result),
            })

    return tasks


def main():
    for p in (INPUT_FILE, STEP11_FILE, FILTER_FILE):
        if not p.exists():
            raise FileNotFoundError(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step11 = load_module(STEP11_FILE, "fh_step11")
    teacher = step11.load_module(step11.SOURCE_STEP3, "fh_teacher")
    filter_v23 = load_module(FILTER_FILE, "fh_filter_v23")

    original_prompt = teacher.build_teacher_user_prompt

    def build_prompt_v4(query_record, selected_result):
        base = original_prompt(
            query_record=query_record,
            selected_result=selected_result,
        )
        return (
            base
            + "\n\nTeacher-v4 reminder: Use only explicit evidence. "
            "Do not add fitness-goal benefits, health effects, qualitative "
            "nutrient judgements, or inferred cuisine facts."
        )

    teacher.SYSTEM_PROMPT = V4_SYSTEM_PROMPT
    teacher.build_teacher_user_prompt = build_prompt_v4

    queries = read_jsonl(INPUT_FILE)
    tasks = build_tasks(teacher, queries)

    target = len(tasks)
    if target != 30000:
        raise RuntimeError(f"Expected 30000 tasks, got {target}")

    existing = set()
    if RAW_FILE.exists():
        existing = {
            str(r.get("sample_id", ""))
            for r in read_jsonl(RAW_FILE)
        }

    pending = [t for t in tasks if t["sample_id"] not in existing]

    print("=" * 72)
    print("Fitness Home - Main-20K Teacher-v4 Generation")
    print("=" * 72)
    print("Queries     :", len(queries))
    print("Raw target  :", target)
    print("Existing    :", len(existing))
    print("Pending     :", len(pending))
    print("Batch size  :", BATCH_SIZE)

    if pending:
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        tokenizer, model = teacher.load_teacher_model()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        start = time.time()
        done = 0

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]

            outputs = step11.generate_batch_safe(
                teacher=teacher,
                tokenizer=tokenizer,
                model=model,
                tasks=batch,
            )

            for task, data in zip(batch, outputs):
                user_prompt, teacher_output, seconds = data

                source_reasons = teacher.validate_output(
                    output=teacher_output,
                    query_record=task["query_record"],
                    result=task["selected_result"],
                )

                record = teacher.build_training_record(
                    task=task,
                    user_prompt=user_prompt,
                    teacher_output=teacher_output,
                    accepted=(len(source_reasons) == 0),
                    rejection_reasons=source_reasons,
                    generation_seconds=seconds,
                )

                md = record.setdefault("metadata", {})
                md["dataset_version"] = "main20k_teacher_v4_raw30k"
                md["constraint_signature_id"] = task[
                    "query_record"
                ].get("constraint_signature_id")
                md["paraphrase_id"] = task[
                    "query_record"
                ].get("paraphrase_id")
                md["source_validator_reasons"] = list(source_reasons)

                v23_reasons = filter_v23.filter_reasons(record)
                md["filter_v2_3_accepted"] = not v23_reasons
                md["filter_v2_3_rejection_reasons"] = v23_reasons

                append_jsonl(RAW_FILE, record)
                done += 1

            total_done = len(existing) + done
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0.0

            if total_done % 100 == 0 or total_done == target:
                print(
                    f"[{total_done:05d}/{target}] "
                    f"rate={rate:.2f}/s"
                )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    raw = read_jsonl(RAW_FILE)

    accepted = [
        r for r in raw
        if r.get("metadata", {}).get("filter_v2_3_accepted")
    ]
    rejected = [
        r for r in raw
        if not r.get("metadata", {}).get("filter_v2_3_accepted")
    ]

    write_jsonl(ACC_FILE, accepted)
    write_jsonl(REJ_FILE, rejected)

    match = Counter(
        str(r.get("metadata", {}).get("match_type", "unknown"))
        for r in accepted
    )

    reasons = Counter()
    for r in rejected:
        for reason in r.get("metadata", {}).get(
            "filter_v2_3_rejection_reasons", []
        ):
            reasons[str(reason).split(":", 1)[0]] += 1

    summary = {
        "dataset_version": "main20k_teacher_v4_raw30k",
        "raw": len(raw),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_rate": len(accepted) / len(raw) if raw else 0.0,
        "accepted_match": dict(match),
        "top_reject_reasons": reasons.most_common(12),
        "unique_signatures": len({
            str(r.get("metadata", {}).get("constraint_signature_id", ""))
            for r in raw
        } - {""}),
        "unique_queries": len({
            str(r.get("metadata", {}).get("query_id", ""))
            for r in raw
        } - {""}),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("MAIN-20K TEACHER-v4 RESULT")
    print("=" * 72)
    print("Raw      :", len(raw))
    print("Accepted :", len(accepted))
    print("Rejected :", len(rejected))
    print("Rate     :", f"{summary['accepted_rate']:.2%}")
    print("Match    :", dict(match))
    print("Top reject reasons:", reasons.most_common(12))
    print("Unique signatures:", summary["unique_signatures"])
    print("Unique queries   :", summary["unique_queries"])
    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
