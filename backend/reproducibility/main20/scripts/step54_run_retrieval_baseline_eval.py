#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parents[2] / "experiments" / "2026_08_week1"
BACKEND = HERE.parents[2]

QUERY_FILE = HERE / "01_query_space" / "queries_main20k.jsonl"
DENSE5_FILE = HERE / "01_query_space" / "retrieval_results_main20k.jsonl"

STEP10_CANDIDATES = [
    WEEK1 / "step10_build_queries_2500_fixed.py",
    WEEK1 / "step10_build_queries_2500.py",
]

OUTPUT_ROOT = HERE / "43_retrieval_baseline_eval"

TOP_K = 5
CANDIDATE_K = 50
SEED = 42
BOOTSTRAP_REPETITIONS = 2000

METHOD_ORDER = (
    "B1_BM25",
    "B2_BM25_ConstraintRerank",
    "B3_DenseBGE",
    "B4_DenseBGE_ConstraintRerank",
    "B5_StructuredConstraintOracle",
)

METHOD_NAMES = {
    "B1_BM25": "BM25@5",
    "B2_BM25_ConstraintRerank": "BM25@50 + Constraint Rerank@5",
    "B3_DenseBGE": "BGE + FAISS Dense@5",
    "B4_DenseBGE_ConstraintRerank": "BGE + FAISS Dense@50 + Constraint Rerank@5",
    "B5_StructuredConstraintOracle": "Structured Constraint Filter@5 (oracle parsed constraints)",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Fitness Home retrieval baselines over the "
            "Main-20K 7,500-query development query space."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test query limit. Full run uses all 7,500 queries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete generated caches/results for this run.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=CANDIDATE_K,
        help="Candidate depth before constraint reranking.",
    )
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def tokenise(value: Any) -> list[str]:
    return TOKEN_RE.findall(normalise_text(value))


def restaurant_id_of(record: dict[str, Any], step10: Any) -> str:
    if hasattr(step10, "restaurant_id_of"):
        return str(step10.restaurant_id_of(record))
    for key in ("restaurant_id", "id", "Restaurant ID", "restaurantId"):
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    database_record = record.get("database_record")
    if isinstance(database_record, dict):
        return restaurant_id_of(database_record, step10)
    raise RuntimeError(
        f"Could not determine restaurant ID from keys: {list(record.keys())}"
    )


def document_text_of(record: dict[str, Any]) -> str:
    for key in ("document_text", "text", "rag_document", "content"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    preferred_keys = (
        "restaurant_name",
        "name",
        "category",
        "cuisine_tags",
        "price_range",
        "address",
        "average_calories",
        "average_protein",
        "average_fiber",
        "average_fibre",
        "health_score",
        "fitness_score",
    )
    parts: list[str] = []
    for key in preferred_keys:
        value = record.get(key)
        if value not in (None, ""):
            parts.append(f"{key.replace('_', ' ')}: {value}")

    if parts:
        return "\n".join(parts)

    return json.dumps(record, ensure_ascii=False, sort_keys=True)


class BM25Index:
    def __init__(
        self,
        documents: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.document_count = len(documents)
        self.document_lengths = np.zeros(self.document_count, dtype=np.float32)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        temporary_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()

        for document_index, document in enumerate(documents):
            counts = Counter(tokenise(document))
            length = sum(counts.values())
            self.document_lengths[document_index] = float(length)

            for term, frequency in counts.items():
                temporary_postings[term].append(
                    (document_index, int(frequency))
                )
                document_frequency[term] += 1

            if (document_index + 1) % 500 == 0:
                print(
                    f"[BM25 index] {document_index + 1}/{self.document_count}",
                    flush=True,
                )

        self.average_document_length = float(
            self.document_lengths.mean()
        ) if self.document_count else 1.0

        self.idf: dict[str, float] = {}
        for term, postings in temporary_postings.items():
            ids = np.asarray(
                [item[0] for item in postings],
                dtype=np.int32,
            )
            frequencies = np.asarray(
                [item[1] for item in postings],
                dtype=np.float32,
            )
            self.postings[term] = (ids, frequencies)

            df = document_frequency[term]
            self.idf[term] = math.log(
                1.0
                + (
                    self.document_count - df + 0.5
                )
                / (df + 0.5)
            )

    def top_indices(
        self,
        query: str,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros(
            self.document_count,
            dtype=np.float32,
        )

        for term in set(tokenise(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue

            document_ids, term_frequencies = posting
            document_lengths = self.document_lengths[document_ids]
            denominator = (
                term_frequencies
                + self.k1
                * (
                    1.0
                    - self.b
                    + self.b
                    * document_lengths
                    / max(
                        self.average_document_length,
                        1e-12,
                    )
                )
            )
            contribution = (
                self.idf[term]
                * (
                    term_frequencies
                    * (self.k1 + 1.0)
                )
                / denominator
            )
            scores[document_ids] += contribution

        requested = min(top_k, self.document_count)
        if requested <= 0:
            return (
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.float32),
            )

        if requested == self.document_count:
            candidate_ids = np.arange(
                self.document_count,
                dtype=np.int32,
            )
        else:
            candidate_ids = np.argpartition(
                scores,
                -requested,
            )[-requested:]

        order = np.lexsort(
            (
                candidate_ids,
                -scores[candidate_ids],
            )
        )
        ranked_ids = candidate_ids[order]
        return ranked_ids, scores[ranked_ids]


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise RuntimeError(
                f"Cache row without query_id: {path}"
            )
        if query_id in result:
            raise RuntimeError(
                f"Duplicate query_id {query_id} in {path}"
            )
        result[query_id] = row
    return result


def check_fraction(
    evaluated_result: dict[str, Any],
    constraints: dict[str, Any],
) -> float:
    checks = evaluated_result.get("constraint_checks", {})
    if not isinstance(checks, dict):
        return 0.0

    required = ["cuisine", "calories", "protein"]
    if constraints.get("min_fiber") is not None:
        required.append(
            "fiber"
            if "fiber" in checks
            else "fibre"
        )

    values = [
        bool(checks.get(name))
        for name in required
    ]
    return sum(values) / len(values) if values else 0.0


def match_priority(result: dict[str, Any]) -> int:
    match_type = str(
        result.get("match_type", "partial")
    ).lower()
    return {
        "full": 2,
        "weak": 1,
        "partial": 0,
    }.get(match_type, 0)


def rerank_by_constraints(
    evaluated_results: Sequence[dict[str, Any]],
    constraints: dict[str, Any],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(item) for item in evaluated_results),
        key=lambda item: (
            -match_priority(item),
            -check_fraction(item, constraints),
            -float(
                item.get(
                    "similarity_score",
                    item.get("bm25_score", 0.0),
                )
                or 0.0
            ),
            str(item.get("restaurant_id", "")),
        ),
    )[:top_k]

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def evaluate_raw_results(
    raw_results: Sequence[dict[str, Any]],
    query_row: dict[str, Any],
    step10: Any,
    step2: Any,
) -> list[dict[str, Any]]:
    relevant_ids = {
        str(value)
        for value in query_row[
            "ground_truth_restaurant_ids"
        ]
    }
    return step10.evaluate_results(
        step2=step2,
        raw_results=list(raw_results),
        constraints=query_row["constraints"],
        relevant_ids=relevant_ids,
    )


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    return sum(
        restaurant_id in relevant_ids
        for restaurant_id in retrieved_ids[:k]
    ) / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0
    return sum(
        restaurant_id in relevant_ids
        for restaurant_id in retrieved_ids[:k]
    ) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    for rank, restaurant_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if restaurant_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    dcg = 0.0
    for index, restaurant_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if restaurant_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 1.0)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(
        1.0 / math.log2(index + 1.0)
        for index in range(1, ideal_hits + 1)
    )
    return dcg / idcg if idcg else 0.0


def per_query_metrics(
    query_row: dict[str, Any],
    results: Sequence[dict[str, Any]],
    step10: Any,
) -> dict[str, Any]:
    relevant_ids = {
        str(value)
        for value in query_row[
            "ground_truth_restaurant_ids"
        ]
    }
    retrieved_ids = [
        restaurant_id_of(dict(result), step10)
        for result in results[:TOP_K]
    ]

    satisfaction = [
        check_fraction(
            result,
            query_row["constraints"],
        )
        for result in results[:TOP_K]
    ]

    full_flags = [
        str(
            result.get("match_type", "partial")
        ).lower()
        == "full"
        for result in results[:TOP_K]
    ]

    return {
        "query_id": query_row["query_id"],
        "constraint_signature_id": query_row[
            "constraint_signature_id"
        ],
        "cuisine": query_row["constraints"].get(
            "cuisine"
        ),
        "goal": query_row["constraints"].get(
            "goal"
        ),
        "fiber_required": (
            query_row["constraints"].get(
                "min_fiber"
            )
            is not None
        ),
        "ground_truth_size": len(relevant_ids),
        "precision_at_5": precision_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        ),
        "recall_at_5": recall_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        ),
        "mrr": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
        "ndcg_at_5": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        ),
        "hit_at_5": any(
            restaurant_id in relevant_ids
            for restaurant_id in retrieved_ids
        ),
        "relevant_count_at_5": sum(
            restaurant_id in relevant_ids
            for restaurant_id in retrieved_ids
        ),
        "top1_constraint_satisfaction": (
            satisfaction[0]
            if satisfaction
            else 0.0
        ),
        "best_at_5_constraint_satisfaction": (
            max(satisfaction)
            if satisfaction
            else 0.0
        ),
        "full_match_at_1": (
            full_flags[0]
            if full_flags
            else False
        ),
        "full_match_at_5": any(full_flags),
        "top1_restaurant_id": (
            retrieved_ids[0]
            if retrieved_ids
            else None
        ),
        "top1_restaurant_name": (
            results[0].get("restaurant_name")
            if results
            else None
        ),
    }


def aggregate_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {"query_count": 0}

    top1_counter = Counter(
        str(row.get("top1_restaurant_name", ""))
        for row in rows
        if row.get("top1_restaurant_name")
    )

    def mean(key: str) -> float:
        return statistics.fmean(
            float(row[key])
            for row in rows
        )

    return {
        "query_count": len(rows),
        "mean_precision_at_5": mean(
            "precision_at_5"
        ),
        "mean_recall_at_5": mean(
            "recall_at_5"
        ),
        "mrr": mean("mrr"),
        "mean_ndcg_at_5": mean(
            "ndcg_at_5"
        ),
        "hit_at_5_rate": mean("hit_at_5"),
        "mean_relevant_count_at_5": mean(
            "relevant_count_at_5"
        ),
        "mean_top1_constraint_satisfaction": mean(
            "top1_constraint_satisfaction"
        ),
        "mean_best_at_5_constraint_satisfaction": mean(
            "best_at_5_constraint_satisfaction"
        ),
        "full_match_at_1_rate": mean(
            "full_match_at_1"
        ),
        "full_match_at_5_rate": mean(
            "full_match_at_5"
        ),
        "unique_top1_restaurants": len(
            top1_counter
        ),
        "most_common_top1": (
            top1_counter.most_common(1)[0]
            if top1_counter
            else None
        ),
        "largest_top1_share": (
            top1_counter.most_common(1)[0][1]
            / len(rows)
            if top1_counter
            else 0.0
        ),
    }


def subgroup_key(row: dict[str, Any]) -> list[tuple[str, str]]:
    ground_truth_size = int(
        row["ground_truth_size"]
    )
    if ground_truth_size <= 5:
        size_bucket = "gt_1_5"
    elif ground_truth_size <= 20:
        size_bucket = "gt_6_20"
    else:
        size_bucket = "gt_21_plus"

    return [
        ("cuisine", str(row["cuisine"])),
        ("goal", str(row["goal"])),
        (
            "fiber",
            "required"
            if row["fiber_required"]
            else "not_required",
        ),
        ("ground_truth_size", size_bucket),
    ]


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError(
            "Paired metric vectors have different lengths."
        )
    if not first:
        return {
            "difference_second_minus_first": 0.0,
            "ci_95": [0.0, 0.0],
            "repetitions": repetitions,
        }

    differences = [
        float(second_value)
        - float(first_value)
        for first_value, second_value in zip(
            first,
            second,
        )
    ]
    observed = statistics.fmean(differences)

    rng = random.Random(seed)
    bootstrap_means: list[float] = []

    for _ in range(repetitions):
        sample = [
            differences[
                rng.randrange(len(differences))
            ]
            for _ in range(len(differences))
        ]
        bootstrap_means.append(
            statistics.fmean(sample)
        )

    bootstrap_means.sort()
    lower = bootstrap_means[
        int(0.025 * (repetitions - 1))
    ]
    upper = bootstrap_means[
        int(0.975 * (repetitions - 1))
    ]

    return {
        "difference_second_minus_first": observed,
        "ci_95": [lower, upper],
        "repetitions": repetitions,
    }


def exact_mcnemar(
    first: Sequence[bool],
    second: Sequence[bool],
) -> dict[str, Any]:
    first_only = sum(
        bool(a) and not bool(b)
        for a, b in zip(first, second)
    )
    second_only = sum(
        bool(b) and not bool(a)
        for a, b in zip(first, second)
    )
    discordant = first_only + second_only

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(
            first_only,
            second_only,
        )
        tail = sum(
            math.comb(discordant, index)
            for index in range(smaller + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)

    return {
        "first_only_success": first_only,
        "second_only_success": second_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def write_main_table(
    path_csv: Path,
    path_md: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    headers = [
        "Method",
        "Precision@5",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Hit@5",
        "Top1 CSR",
        "Best@5 CSR",
        "Full@1",
        "Full@5",
        "Unique Top1",
        "Largest Top1 Share",
    ]

    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        summary = summaries[method_id]
        rows.append({
            "Method": (
                f"{method_id} {METHOD_NAMES[method_id]}"
            ),
            "Precision@5": summary[
                "mean_precision_at_5"
            ],
            "Recall@5": summary[
                "mean_recall_at_5"
            ],
            "MRR": summary["mrr"],
            "nDCG@5": summary[
                "mean_ndcg_at_5"
            ],
            "Hit@5": summary[
                "hit_at_5_rate"
            ],
            "Top1 CSR": summary[
                "mean_top1_constraint_satisfaction"
            ],
            "Best@5 CSR": summary[
                "mean_best_at_5_constraint_satisfaction"
            ],
            "Full@1": summary[
                "full_match_at_1_rate"
            ],
            "Full@5": summary[
                "full_match_at_5_rate"
            ],
            "Unique Top1": summary[
                "unique_top1_restaurants"
            ],
            "Largest Top1 Share": summary[
                "largest_top1_share"
            ],
        })

    with path_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(["---"] * len(headers))
        + " |",
    ]

    for row in rows:
        formatted: list[str] = []
        for header in headers:
            value = row[header]
            if header in (
                "Unique Top1",
                "Method",
            ):
                formatted.append(str(value))
            else:
                formatted.append(
                    f"{float(value):.4f}"
                )
        markdown_rows.append(
            "| " + " | ".join(formatted) + " |"
        )

    path_md.write_text(
        "\n".join(markdown_rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.candidate_k < TOP_K:
        raise ValueError(
            "--candidate-k must be >= 5."
        )

    required = (
        QUERY_FILE,
        DENSE5_FILE,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    step10_file = next(
        (
            path
            for path in STEP10_CANDIDATES
            if path.exists()
        ),
        None,
    )
    if step10_file is None:
        raise FileNotFoundError(
            "Could not find Step-10 query builder."
        )

    step10 = load_module(
        step10_file,
        "fh_retrieval_step10",
    )
    step2 = step10.load_module(
        step10.SOURCE_STEP2,
        "fh_retrieval_step2",
    )

    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))

    from rag.retriever import retrieve

    all_queries = read_jsonl(QUERY_FILE)
    dense5_rows = read_jsonl(DENSE5_FILE)

    if len(all_queries) != 7500:
        raise RuntimeError(
            f"Expected 7,500 queries, found {len(all_queries)}."
        )
    if len(dense5_rows) != 7500:
        raise RuntimeError(
            f"Expected 7,500 existing dense results, found {len(dense5_rows)}."
        )

    dense5_by_id = {
        str(row["query_id"]): row
        for row in dense5_rows
    }
    query_ids = [
        str(row["query_id"])
        for row in all_queries
    ]
    if set(query_ids) != set(dense5_by_id):
        raise RuntimeError(
            "Query IDs do not match existing dense retrieval results."
        )

    queries = (
        all_queries[: args.limit]
        if args.limit is not None
        else all_queries
    )

    run_name = (
        f"smoke_{len(queries)}"
        if args.limit is not None
        else "development_7500"
    )
    run_dir = OUTPUT_ROOT / run_name

    if args.overwrite and run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    protocol = {
        "experiment": "fitness_home_retrieval_baseline_v1",
        "status": "frozen_before_result_inspection",
        "development_only": True,
        "blind_test_used": False,
        "query_file": str(QUERY_FILE),
        "query_file_sha256": sha256_file(QUERY_FILE),
        "existing_dense5_file": str(DENSE5_FILE),
        "existing_dense5_sha256": sha256_file(DENSE5_FILE),
        "query_count": len(queries),
        "top_k": TOP_K,
        "candidate_k": args.candidate_k,
        "methods": {
            method_id: METHOD_NAMES[method_id]
            for method_id in METHOD_ORDER
        },
        "metrics": [
            "Precision@5",
            "Recall@5",
            "MRR",
            "nDCG@5",
            "Hit@5",
            "Top1 Constraint Satisfaction Rate",
            "Best@5 Constraint Satisfaction Rate",
            "Full Match@1",
            "Full Match@5",
            "Top1 diversity",
        ],
        "important_note": (
            "B5 receives already parsed structured constraints and acts as a "
            "structured constraint ceiling, not a fair natural-language "
            "retrieval baseline."
        ),
    }
    protocol_path = run_dir / "retrieval_protocol.json"
    protocol_path.write_text(
        json.dumps(
            protocol,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("FITNESS HOME — RETRIEVAL BASELINE EVALUATION")
    print("=" * 78)
    print("Queries             :", len(queries))
    print("Top K               :", TOP_K)
    print("Candidate K         :", args.candidate_k)
    print("Development only    : YES")
    print("Blind test used     : NO")
    print("Output              :", run_dir)

    restaurants = step2.load_restaurants()
    if len(restaurants) != 4996:
        raise RuntimeError(
            f"Expected 4,996 restaurants, found {len(restaurants)}."
        )

    restaurant_ids = [
        restaurant_id_of(
            dict(record),
            step10,
        )
        for record in restaurants
    ]
    if len(restaurant_ids) != len(
        set(restaurant_ids)
    ):
        raise RuntimeError(
            "Duplicate restaurant IDs in retrieval corpus."
        )

    corpus_texts = [
        document_text_of(dict(record))
        for record in restaurants
    ]

    print("Building BM25 index...")
    bm25 = BM25Index(corpus_texts)
    print(
        "BM25 vocabulary      :",
        len(bm25.postings),
    )

    bm25_cache_path = (
        run_dir / "bm25_top50_results.jsonl"
    )
    dense50_cache_path = (
        run_dir / "dense_top50_results.jsonl"
    )
    structured_cache_path = (
        run_dir
        / "structured_results_by_signature.jsonl"
    )

    bm25_cache = load_completed(
        bm25_cache_path
    )
    dense50_cache = load_completed(
        dense50_cache_path
    )
    structured_cache = load_completed(
        structured_cache_path
    )

    # B5 is constant across paraphrases with the same signature.
    queries_by_signature: dict[
        str,
        dict[str, Any],
    ] = {}
    for query in queries:
        queries_by_signature.setdefault(
            str(
                query[
                    "constraint_signature_id"
                ]
            ),
            query,
        )

    for signature_index, (
        signature_id,
        representative_query,
    ) in enumerate(
        queries_by_signature.items(),
        start=1,
    ):
        if signature_id in structured_cache:
            continue

        evaluated = evaluate_raw_results(
            restaurants,
            representative_query,
            step10,
            step2,
        )
        ranked = rerank_by_constraints(
            evaluated,
            representative_query[
                "constraints"
            ],
            top_k=TOP_K,
        )
        row = {
            "query_id": signature_id,
            "constraint_signature_id": (
                signature_id
            ),
            "retrieval_results": ranked,
        }
        append_jsonl(
            structured_cache_path,
            row,
        )
        structured_cache[
            signature_id
        ] = row

        if (
            signature_index % 50 == 0
            or signature_index
            == len(queries_by_signature)
        ):
            print(
                f"[B5 structured] "
                f"{signature_index}/"
                f"{len(queries_by_signature)}",
                flush=True,
            )

    for index, query in enumerate(
        queries,
        start=1,
    ):
        query_id = str(query["query_id"])

        if query_id not in bm25_cache:
            document_indices, scores = (
                bm25.top_indices(
                    str(query["query"]),
                    args.candidate_k,
                )
            )
            raw_results: list[dict[str, Any]] = []
            for rank, (
                document_index,
                score,
            ) in enumerate(
                zip(
                    document_indices,
                    scores,
                ),
                start=1,
            ):
                result = dict(
                    restaurants[
                        int(document_index)
                    ]
                )
                result["rank"] = rank
                result["bm25_score"] = float(
                    score
                )
                result["similarity_score"] = float(
                    score
                )
                raw_results.append(result)

            evaluated = evaluate_raw_results(
                raw_results,
                query,
                step10,
                step2,
            )
            row = {
                **query,
                "retrieval_results": evaluated,
            }
            append_jsonl(
                bm25_cache_path,
                row,
            )
            bm25_cache[query_id] = row

        if query_id not in dense50_cache:
            raw_dense = retrieve(
                str(query["query"]),
                top_k=args.candidate_k,
            )
            evaluated_dense = (
                evaluate_raw_results(
                    raw_dense,
                    query,
                    step10,
                    step2,
                )
            )
            row = {
                **query,
                "retrieval_results": (
                    evaluated_dense
                ),
            }
            append_jsonl(
                dense50_cache_path,
                row,
            )
            dense50_cache[query_id] = row

        if index % 100 == 0 or index == len(
            queries
        ):
            print(
                f"[query retrieval] "
                f"{index}/{len(queries)}",
                flush=True,
            )

    method_per_query: dict[
        str,
        list[dict[str, Any]],
    ] = {
        method_id: []
        for method_id in METHOD_ORDER
    }

    per_query_path = (
        run_dir / "retrieval_per_query_metrics.jsonl"
    )
    if per_query_path.exists():
        per_query_path.unlink()

    for query in queries:
        query_id = str(query["query_id"])
        signature_id = str(
            query[
                "constraint_signature_id"
            ]
        )

        dense5_results = (
            dense5_by_id[query_id].get(
                "retrieval_results",
                [],
            )[:TOP_K]
        )
        bm25_50_results = (
            bm25_cache[query_id].get(
                "retrieval_results",
                [],
            )
        )
        dense50_results = (
            dense50_cache[query_id].get(
                "retrieval_results",
                [],
            )
        )

        method_results = {
            "B1_BM25": (
                bm25_50_results[:TOP_K]
            ),
            "B2_BM25_ConstraintRerank": (
                rerank_by_constraints(
                    bm25_50_results,
                    query["constraints"],
                    top_k=TOP_K,
                )
            ),
            "B3_DenseBGE": dense5_results,
            "B4_DenseBGE_ConstraintRerank": (
                rerank_by_constraints(
                    dense50_results,
                    query["constraints"],
                    top_k=TOP_K,
                )
            ),
            "B5_StructuredConstraintOracle": (
                structured_cache[
                    signature_id
                ]["retrieval_results"]
            ),
        }

        output_row = {
            "query_id": query_id,
            "constraint_signature_id": (
                signature_id
            ),
            "query": query["query"],
            "constraints": query[
                "constraints"
            ],
            "methods": {},
        }

        for method_id, results in (
            method_results.items()
        ):
            metrics = per_query_metrics(
                query,
                results,
                step10,
            )
            method_per_query[
                method_id
            ].append(metrics)
            output_row["methods"][
                method_id
            ] = {
                "metrics": metrics,
                "retrieval_results": results,
            }

        append_jsonl(
            per_query_path,
            output_row,
        )

    summaries = {
        method_id: aggregate_metrics(rows)
        for method_id, rows in (
            method_per_query.items()
        )
    }

    subgroup_rows: list[dict[str, Any]] = []
    for method_id, rows in (
        method_per_query.items()
    ):
        grouped: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = defaultdict(list)
        for row in rows:
            for group_type, group_value in (
                subgroup_key(row)
            ):
                grouped[
                    (
                        group_type,
                        group_value,
                    )
                ].append(row)

        for (
            group_type,
            group_value,
        ), group_rows in sorted(
            grouped.items()
        ):
            subgroup_rows.append({
                "method_id": method_id,
                "method_name": (
                    METHOD_NAMES[method_id]
                ),
                "group_type": group_type,
                "group_value": group_value,
                **aggregate_metrics(
                    group_rows
                ),
            })

    subgroup_path = (
        run_dir
        / "retrieval_subgroup_table.csv"
    )
    subgroup_headers = list(
        subgroup_rows[0].keys()
    )
    with subgroup_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=subgroup_headers,
        )
        writer.writeheader()
        writer.writerows(subgroup_rows)

    main_csv = (
        run_dir
        / "retrieval_main_table.csv"
    )
    main_md = (
        run_dir
        / "retrieval_main_table.md"
    )
    write_main_table(
        main_csv,
        main_md,
        summaries,
    )

    pairs = (
        (
            "B1_BM25",
            "B2_BM25_ConstraintRerank",
        ),
        (
            "B3_DenseBGE",
            "B4_DenseBGE_ConstraintRerank",
        ),
        (
            "B2_BM25_ConstraintRerank",
            "B4_DenseBGE_ConstraintRerank",
        ),
    )

    significance: list[
        dict[str, Any]
    ] = []
    for first_id, second_id in pairs:
        first_rows = method_per_query[
            first_id
        ]
        second_rows = method_per_query[
            second_id
        ]

        significance.append({
            "comparison": (
                f"{first_id}_vs_{second_id}"
            ),
            "direction": (
                f"All differences are "
                f"{second_id} minus {first_id}."
            ),
            "recall_at_5": paired_bootstrap(
                [
                    row["recall_at_5"]
                    for row in first_rows
                ],
                [
                    row["recall_at_5"]
                    for row in second_rows
                ],
                seed=SEED,
            ),
            "ndcg_at_5": paired_bootstrap(
                [
                    row["ndcg_at_5"]
                    for row in first_rows
                ],
                [
                    row["ndcg_at_5"]
                    for row in second_rows
                ],
                seed=SEED + 1,
            ),
            "top1_constraint_satisfaction": paired_bootstrap(
                [
                    row[
                        "top1_constraint_satisfaction"
                    ]
                    for row in first_rows
                ],
                [
                    row[
                        "top1_constraint_satisfaction"
                    ]
                    for row in second_rows
                ],
                seed=SEED + 2,
            ),
            "hit_at_5": exact_mcnemar(
                [
                    row["hit_at_5"]
                    for row in first_rows
                ],
                [
                    row["hit_at_5"]
                    for row in second_rows
                ],
            ),
            "full_match_at_5": exact_mcnemar(
                [
                    row[
                        "full_match_at_5"
                    ]
                    for row in first_rows
                ],
                [
                    row[
                        "full_match_at_5"
                    ]
                    for row in second_rows
                ],
            ),
        })

    significance_path = (
        run_dir
        / "retrieval_pairwise_significance.json"
    )
    significance_path.write_text(
        json.dumps(
            significance,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = (
        run_dir
        / "retrieval_evaluation_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            {
                "experiment": (
                    "fitness_home_retrieval_baseline_v1"
                ),
                "development_only": True,
                "blind_test_used": False,
                "query_count": len(queries),
                "top_k": TOP_K,
                "candidate_k": args.candidate_k,
                "methods": summaries,
                "structured_oracle_note": (
                    "B5 receives frozen parsed constraints "
                    "and is reported as a structured ceiling."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checksum_path = (
        run_dir / "SHA256SUMS.txt"
    )
    output_files = (
        protocol_path,
        bm25_cache_path,
        dense50_cache_path,
        structured_cache_path,
        per_query_path,
        subgroup_path,
        main_csv,
        main_md,
        significance_path,
        summary_path,
    )
    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for path in output_files:
            file.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    print()
    print("=" * 78)
    print("RETRIEVAL BASELINE EVALUATION COMPLETE")
    print("=" * 78)
    for method_id in METHOD_ORDER:
        summary = summaries[method_id]
        print(
            f"{method_id} "
            f"P@5={summary['mean_precision_at_5']:.4f} "
            f"R@5={summary['mean_recall_at_5']:.4f} "
            f"MRR={summary['mrr']:.4f} "
            f"nDCG@5={summary['mean_ndcg_at_5']:.4f} "
            f"Hit@5={summary['hit_at_5_rate']:.2%} "
            f"Top1CSR={summary['mean_top1_constraint_satisfaction']:.2%} "
            f"Full@5={summary['full_match_at_5_rate']:.2%}"
        )
    print("Main table       :", main_csv)
    print("Markdown table   :", main_md)
    print("Subgroup table   :", subgroup_path)
    print("Significance     :", significance_path)
    print("Summary          :", summary_path)
    print("Blind test used  : NO")


if __name__ == "__main__":
    main()
