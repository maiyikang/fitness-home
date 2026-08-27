#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent

STEP54_FILE = HERE / "step54_run_retrieval_baseline_eval.py"
QUERY_FILE = HERE / "01_query_space" / "queries_main20k.jsonl"

FROZEN_BASELINE_DIR = (
    HERE
    / "43_retrieval_baseline_eval"
    / "development_7500_frozen"
)
LIVE_BASELINE_DIR = (
    HERE
    / "43_retrieval_baseline_eval"
    / "development_7500"
)

OUT_DIR = HERE / "44_hybrid_retrieval_postprocess" / "development_7500"

TOP_K = 5
SOURCE_CANDIDATE_K = 50
FUSED_CANDIDATE_K = 50
RRF_K = 60
SEED = 42
BOOTSTRAP_REPETITIONS = 2000

BASE_METHOD_ORDER = (
    "B1_BM25",
    "B2_BM25_ConstraintRerank",
    "B3_DenseBGE",
    "B4_DenseBGE_ConstraintRerank",
    "B5_StructuredConstraintOracle",
)
HYBRID_METHOD_ID = "B6_HybridRRF_ConstraintRerank"

METHOD_NAMES = {
    "B1_BM25": "BM25@5",
    "B2_BM25_ConstraintRerank": "BM25@50 + Constraint Rerank@5",
    "B3_DenseBGE": "BGE + FAISS Dense@5",
    "B4_DenseBGE_ConstraintRerank": (
        "BGE + FAISS Dense@50 + Constraint Rerank@5"
    ),
    "B5_StructuredConstraintOracle": (
        "Structured Constraint Filter@5 (oracle parsed constraints)"
    ),
    HYBRID_METHOD_ID: (
        "BM25@50 ∪ Dense@50 → RRF@50 → Constraint Rerank@5"
    ),
}


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_sha256sums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            digest, filename = line.split(maxsplit=1)
            values[filename.strip()] = digest.strip()
    return values


def verify_frozen_inputs(base_dir: Path) -> None:
    checksum_file = base_dir / "SHA256SUMS.txt"
    if not checksum_file.exists():
        raise FileNotFoundError(checksum_file)

    expected = parse_sha256sums(checksum_file)
    required = (
        "bm25_top50_results.jsonl",
        "dense_top50_results.jsonl",
        "retrieval_per_query_metrics.jsonl",
        "retrieval_protocol.json",
    )

    for filename in required:
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        if filename not in expected:
            raise RuntimeError(
                f"{filename} is absent from frozen SHA256SUMS.txt"
            )
        observed = sha256_file(path)
        if observed != expected[filename]:
            raise RuntimeError(
                f"Frozen checksum mismatch for {filename}: "
                f"{observed} != {expected[filename]}"
            )


def restaurant_id_of(
    result: dict[str, Any],
    step54: Any,
    step10: Any,
) -> str:
    return str(step54.restaurant_id_of(result, step10))


def complete_result_score(result: dict[str, Any]) -> int:
    keys = (
        "constraint_checks",
        "match_type",
        "restaurant_name",
        "restaurant_id",
        "database_record",
    )
    return sum(key in result for key in keys)


def merge_candidate_records(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if current is None:
        return dict(candidate)

    # Keep the richer evaluated record, then fill missing fields from the other.
    if complete_result_score(candidate) > complete_result_score(current):
        merged = dict(candidate)
        fallback = current
    else:
        merged = dict(current)
        fallback = candidate

    for key, value in fallback.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value

    return merged


def build_rrf_candidates(
    bm25_results: Sequence[dict[str, Any]],
    dense_results: Sequence[dict[str, Any]],
    step54: Any,
    step10: Any,
) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}

    for source_name, results in (
        ("bm25", bm25_results[:SOURCE_CANDIDATE_K]),
        ("dense", dense_results[:SOURCE_CANDIDATE_K]),
    ):
        for rank, raw_result in enumerate(results, start=1):
            result = dict(raw_result)
            restaurant_id = restaurant_id_of(
                result,
                step54,
                step10,
            )

            entry = combined.setdefault(
                restaurant_id,
                {
                    "restaurant_id": restaurant_id,
                    "record": None,
                    "bm25_rank": None,
                    "dense_rank": None,
                    "rrf_score": 0.0,
                },
            )

            entry["record"] = merge_candidate_records(
                entry["record"],
                result,
            )
            entry[f"{source_name}_rank"] = rank
            entry["rrf_score"] += 1.0 / (RRF_K + rank)

    ranked_entries = sorted(
        combined.values(),
        key=lambda entry: (
            -float(entry["rrf_score"]),
            min(
                rank
                for rank in (
                    entry["bm25_rank"],
                    entry["dense_rank"],
                )
                if rank is not None
            ),
            str(entry["restaurant_id"]),
        ),
    )[:FUSED_CANDIDATE_K]

    fused: list[dict[str, Any]] = []

    for fused_rank, entry in enumerate(ranked_entries, start=1):
        result = dict(entry["record"] or {})
        result["restaurant_id"] = entry["restaurant_id"]
        result["bm25_rank"] = entry["bm25_rank"]
        result["dense_rank"] = entry["dense_rank"]
        result["rrf_score"] = float(entry["rrf_score"])
        result["similarity_score"] = float(entry["rrf_score"])
        result["rank"] = fused_rank
        fused.append(result)

    return fused


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise RuntimeError(
                f"Hybrid cache row without query_id: {path}"
            )
        if query_id in by_id:
            raise RuntimeError(
                f"Duplicate query_id in hybrid cache: {query_id}"
            )
        by_id[query_id] = row

    return by_id


def aggregate_with_id_diversity(
    rows: Sequence[dict[str, Any]],
    step54: Any,
) -> dict[str, Any]:
    summary = dict(step54.aggregate_metrics(rows))

    top1_ids = [
        str(row["top1_restaurant_id"])
        for row in rows
        if row.get("top1_restaurant_id") not in (None, "")
    ]
    counter = Counter(top1_ids)

    summary["unique_top1_restaurants"] = len(counter)
    summary["most_common_top1"] = (
        counter.most_common(1)[0]
        if counter
        else None
    )
    summary["largest_top1_share"] = (
        counter.most_common(1)[0][1] / len(rows)
        if counter
        else 0.0
    )
    summary["diversity_key"] = "restaurant_id"

    return summary


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    return step54_module.paired_bootstrap(
        first,
        second,
        seed=seed,
        repetitions=BOOTSTRAP_REPETITIONS,
    )


def exact_mcnemar(
    first: Sequence[bool],
    second: Sequence[bool],
) -> dict[str, Any]:
    return step54_module.exact_mcnemar(first, second)


def compare_methods(
    first_id: str,
    second_id: str,
    method_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    first = method_rows[first_id]
    second = method_rows[second_id]

    if len(first) != len(second):
        raise RuntimeError(
            f"Paired length mismatch: {first_id} vs {second_id}"
        )

    return {
        "comparison": f"{first_id}_vs_{second_id}",
        "direction": (
            f"All differences are {second_id} minus {first_id}."
        ),
        "recall_at_5": paired_bootstrap(
            [row["recall_at_5"] for row in first],
            [row["recall_at_5"] for row in second],
            seed=SEED,
        ),
        "ndcg_at_5": paired_bootstrap(
            [row["ndcg_at_5"] for row in first],
            [row["ndcg_at_5"] for row in second],
            seed=SEED + 1,
        ),
        "top1_constraint_satisfaction": paired_bootstrap(
            [
                row["top1_constraint_satisfaction"]
                for row in first
            ],
            [
                row["top1_constraint_satisfaction"]
                for row in second
            ],
            seed=SEED + 2,
        ),
        "hit_at_5": exact_mcnemar(
            [bool(row["hit_at_5"]) for row in first],
            [bool(row["hit_at_5"]) for row in second],
        ),
        "full_match_at_5": exact_mcnemar(
            [
                bool(row["full_match_at_5"])
                for row in first
            ],
            [
                bool(row["full_match_at_5"])
                for row in second
            ],
        ),
    }


def write_main_table(
    path_csv: Path,
    path_md: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    method_order = (*BASE_METHOD_ORDER, HYBRID_METHOD_ID)

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

    for method_id in method_order:
        summary = summaries[method_id]
        rows.append({
            "Method": f"{method_id} {METHOD_NAMES[method_id]}",
            "Precision@5": summary["mean_precision_at_5"],
            "Recall@5": summary["mean_recall_at_5"],
            "MRR": summary["mrr"],
            "nDCG@5": summary["mean_ndcg_at_5"],
            "Hit@5": summary["hit_at_5_rate"],
            "Top1 CSR": summary[
                "mean_top1_constraint_satisfaction"
            ],
            "Best@5 CSR": summary[
                "mean_best_at_5_constraint_satisfaction"
            ],
            "Full@1": summary["full_match_at_1_rate"],
            "Full@5": summary["full_match_at_5_rate"],
            "Unique Top1": summary["unique_top1_restaurants"],
            "Largest Top1 Share": summary["largest_top1_share"],
        })

    with path_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values: list[str] = []
        for header in headers:
            value = row[header]
            if header == "Method":
                values.append(str(value))
            elif header == "Unique Top1":
                values.append(str(int(value)))
            else:
                values.append(f"{float(value):.4f}")
        markdown.append("| " + " | ".join(values) + " |")

    path_md.write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


def decision_from_comparison(
    b2_summary: dict[str, Any],
    b6_summary: dict[str, Any],
    b2_vs_b6: dict[str, Any],
) -> dict[str, Any]:
    recall = b2_vs_b6["recall_at_5"]
    ndcg = b2_vs_b6["ndcg_at_5"]
    csr = b2_vs_b6["top1_constraint_satisfaction"]
    hit = b2_vs_b6["hit_at_5"]

    recall_significantly_higher = recall["ci_95"][0] > 0.0
    ndcg_significantly_higher = ndcg["ci_95"][0] > 0.0
    hit_significantly_higher = (
        b6_summary["hit_at_5_rate"]
        > b2_summary["hit_at_5_rate"]
        and hit["two_sided_exact_p"] < 0.05
    )
    top1_csr_not_lower = (
        b6_summary["mean_top1_constraint_satisfaction"]
        >= b2_summary["mean_top1_constraint_satisfaction"]
    )

    recommend_hybrid = bool(
        recall_significantly_higher
        and ndcg_significantly_higher
        and hit_significantly_higher
        and top1_csr_not_lower
    )

    return {
        "frozen_decision_rule": (
            "Recommend Hybrid only if it significantly improves "
            "Recall@5, nDCG@5, and Hit@5 over B2, while observed "
            "Top-1 Constraint Satisfaction is not lower than B2."
        ),
        "criteria": {
            "recall_significantly_higher": (
                recall_significantly_higher
            ),
            "ndcg_significantly_higher": (
                ndcg_significantly_higher
            ),
            "hit_significantly_higher": (
                hit_significantly_higher
            ),
            "top1_csr_not_lower": (
                top1_csr_not_lower
            ),
        },
        "recommended_final_retriever": (
            HYBRID_METHOD_ID
            if recommend_hybrid
            else "B2_BM25_ConstraintRerank"
        ),
        "hybrid_selected": recommend_hybrid,
        "important_note": (
            "This is a single frozen fusion test using RRF k=60 and "
            "a fixed fused candidate budget of 50. No fusion weight "
            "or RRF parameter sweep is permitted after result inspection."
        ),
    }


def main() -> None:
    global step54_module

    if not STEP54_FILE.exists():
        raise FileNotFoundError(STEP54_FILE)
    if not QUERY_FILE.exists():
        raise FileNotFoundError(QUERY_FILE)

    base_dir = (
        FROZEN_BASELINE_DIR
        if FROZEN_BASELINE_DIR.exists()
        else LIVE_BASELINE_DIR
    )
    verify_frozen_inputs(base_dir)

    step54_module = load_module(
        STEP54_FILE,
        "fh_step54_hybrid_postprocess",
    )

    step10_file = next(
        (
            path
            for path in step54_module.STEP10_CANDIDATES
            if path.exists()
        ),
        None,
    )
    if step10_file is None:
        raise FileNotFoundError(
            "Could not locate the frozen Step-10 query builder."
        )

    step10 = load_module(
        step10_file,
        "fh_step10_hybrid_postprocess",
    )

    queries = read_jsonl(QUERY_FILE)
    bm25_rows = read_jsonl(
        base_dir / "bm25_top50_results.jsonl"
    )
    dense_rows = read_jsonl(
        base_dir / "dense_top50_results.jsonl"
    )
    baseline_rows = read_jsonl(
        base_dir / "retrieval_per_query_metrics.jsonl"
    )

    if not (
        len(queries)
        == len(bm25_rows)
        == len(dense_rows)
        == len(baseline_rows)
        == 7500
    ):
        raise RuntimeError(
            "Expected 7,500 rows in queries and all frozen caches."
        )

    query_by_id = {
        str(row["query_id"]): row
        for row in queries
    }
    bm25_by_id = {
        str(row["query_id"]): row
        for row in bm25_rows
    }
    dense_by_id = {
        str(row["query_id"]): row
        for row in dense_rows
    }
    baseline_by_id = {
        str(row["query_id"]): row
        for row in baseline_rows
    }

    expected_ids = set(query_by_id)
    for label, mapping in (
        ("bm25", bm25_by_id),
        ("dense", dense_by_id),
        ("baseline", baseline_by_id),
    ):
        if set(mapping) != expected_ids:
            raise RuntimeError(
                f"{label} query IDs do not match the frozen query set."
            )

    protocol = read_json(
        base_dir / "retrieval_protocol.json"
    )
    if protocol.get("blind_test_used") is not False:
        raise RuntimeError(
            "Frozen retrieval protocol unexpectedly used blind data."
        )
    if sha256_file(QUERY_FILE) != protocol["query_file_sha256"]:
        raise RuntimeError(
            "Query file checksum does not match the frozen protocol."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hybrid_cache_path = (
        OUT_DIR / "hybrid_rrf_per_query_results.jsonl"
    )
    completed = load_completed(hybrid_cache_path)

    print("=" * 78)
    print("FITNESS HOME — SINGLE HYBRID RETRIEVAL DECISION TEST")
    print("=" * 78)
    print("Queries                  : 7500")
    print("BM25 candidates/query    : 50")
    print("Dense candidates/query   : 50")
    print("RRF constant k           :", RRF_K)
    print("Fused candidates/query   :", FUSED_CANDIDATE_K)
    print("Final Top K              :", TOP_K)
    print("Existing completed       :", len(completed))
    print("Development only         : YES")
    print("Blind test used          : NO")
    print("Output                   :", OUT_DIR)

    hybrid_metrics: list[dict[str, Any]] = []

    for index, query_id in enumerate(
        sorted(expected_ids),
        start=1,
    ):
        if query_id in completed:
            row = completed[query_id]
            hybrid_metrics.append(row["metrics"])
            continue

        query = query_by_id[query_id]
        bm25_results = bm25_by_id[query_id][
            "retrieval_results"
        ]
        dense_results = dense_by_id[query_id][
            "retrieval_results"
        ]

        fused_candidates = build_rrf_candidates(
            bm25_results,
            dense_results,
            step54_module,
            step10,
        )
        hybrid_results = (
            step54_module.rerank_by_constraints(
                fused_candidates,
                query["constraints"],
                top_k=TOP_K,
            )
        )
        metrics = step54_module.per_query_metrics(
            query,
            hybrid_results,
            step10,
        )

        output_row = {
            "query_id": query_id,
            "constraint_signature_id": query[
                "constraint_signature_id"
            ],
            "query": query["query"],
            "constraints": query["constraints"],
            "method_id": HYBRID_METHOD_ID,
            "rrf_k": RRF_K,
            "source_candidate_k": SOURCE_CANDIDATE_K,
            "fused_candidate_k": FUSED_CANDIDATE_K,
            "metrics": metrics,
            "retrieval_results": hybrid_results,
        }
        append_jsonl(hybrid_cache_path, output_row)
        completed[query_id] = output_row
        hybrid_metrics.append(metrics)

        if index % 250 == 0 or index == 7500:
            print(
                f"[hybrid postprocess] {index}/7500",
                flush=True,
            )

    # Rebuild in exact frozen query order after any resume.
    hybrid_metrics = [
        completed[str(query["query_id"])]["metrics"]
        for query in queries
    ]

    baseline_method_metrics: dict[
        str,
        list[dict[str, Any]],
    ] = {
        method_id: []
        for method_id in BASE_METHOD_ORDER
    }

    for query in queries:
        query_id = str(query["query_id"])
        method_data = baseline_by_id[query_id]["methods"]
        for method_id in BASE_METHOD_ORDER:
            baseline_method_metrics[method_id].append(
                method_data[method_id]["metrics"]
            )

    method_metrics = {
        **baseline_method_metrics,
        HYBRID_METHOD_ID: hybrid_metrics,
    }

    summaries = {
        method_id: aggregate_with_id_diversity(
            rows,
            step54_module,
        )
        for method_id, rows in method_metrics.items()
    }

    comparisons = [
        compare_methods(
            "B2_BM25_ConstraintRerank",
            HYBRID_METHOD_ID,
            method_metrics,
        ),
        compare_methods(
            "B4_DenseBGE_ConstraintRerank",
            HYBRID_METHOD_ID,
            method_metrics,
        ),
    ]

    comparison_by_name = {
        comparison["comparison"]: comparison
        for comparison in comparisons
    }
    b2_vs_b6 = comparison_by_name[
        f"B2_BM25_ConstraintRerank_vs_{HYBRID_METHOD_ID}"
    ]

    decision = decision_from_comparison(
        summaries["B2_BM25_ConstraintRerank"],
        summaries[HYBRID_METHOD_ID],
        b2_vs_b6,
    )

    main_csv = OUT_DIR / "hybrid_retrieval_main_table.csv"
    main_md = OUT_DIR / "hybrid_retrieval_main_table.md"
    write_main_table(
        main_csv,
        main_md,
        summaries,
    )

    significance_path = (
        OUT_DIR / "hybrid_pairwise_significance.json"
    )
    significance_path.write_text(
        json.dumps(
            comparisons,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    decision_path = OUT_DIR / "final_retriever_decision.json"
    decision_path.write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = OUT_DIR / "hybrid_retrieval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": (
                    "fitness_home_single_hybrid_rrf_decision_v1"
                ),
                "development_only": True,
                "blind_test_used": False,
                "query_count": 7500,
                "rrf_k": RRF_K,
                "source_candidate_k": SOURCE_CANDIDATE_K,
                "fused_candidate_k": FUSED_CANDIDATE_K,
                "top_k": TOP_K,
                "methods": summaries,
                "decision": decision,
                "frozen_input_directory": str(base_dir),
                "frozen_input_checksums_verified": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    protocol_path = OUT_DIR / "hybrid_protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "status": "frozen_single_test_completed",
                "development_only": True,
                "blind_test_used": False,
                "fusion": "Reciprocal Rank Fusion",
                "rrf_k": RRF_K,
                "bm25_candidate_k": SOURCE_CANDIDATE_K,
                "dense_candidate_k": SOURCE_CANDIDATE_K,
                "fused_candidate_k": FUSED_CANDIDATE_K,
                "constraint_rerank_top_k": TOP_K,
                "parameter_sweep": False,
                "primary_comparison": (
                    f"B2_BM25_ConstraintRerank vs "
                    f"{HYBRID_METHOD_ID}"
                ),
                "decision_rule": decision[
                    "frozen_decision_rule"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_files = (
        hybrid_cache_path,
        main_csv,
        main_md,
        significance_path,
        decision_path,
        summary_path,
        protocol_path,
    )
    checksum_path = OUT_DIR / "SHA256SUMS.txt"
    with checksum_path.open("w", encoding="utf-8") as file:
        for path in output_files:
            file.write(
                f"{sha256_file(path)}  {path.name}\n"
            )

    print()
    print("=" * 78)
    print("HYBRID RETRIEVAL DECISION TEST COMPLETE")
    print("=" * 78)

    for method_id in (
        "B2_BM25_ConstraintRerank",
        "B4_DenseBGE_ConstraintRerank",
        HYBRID_METHOD_ID,
    ):
        summary = summaries[method_id]
        print(
            f"{method_id} "
            f"P@5={summary['mean_precision_at_5']:.4f} "
            f"R@5={summary['mean_recall_at_5']:.4f} "
            f"MRR={summary['mrr']:.4f} "
            f"nDCG@5={summary['mean_ndcg_at_5']:.4f} "
            f"Hit@5={summary['hit_at_5_rate']:.2%} "
            f"Top1CSR={summary['mean_top1_constraint_satisfaction']:.2%} "
            f"UniqueTop1={summary['unique_top1_restaurants']} "
            f"MaxShare={summary['largest_top1_share']:.2%}"
        )

    print(
        "Recommended retriever :",
        decision["recommended_final_retriever"],
    )
    print("Decision criteria     :", decision["criteria"])
    print("Main table            :", main_md)
    print("Significance          :", significance_path)
    print("Decision              :", decision_path)
    print("Blind test used       : NO")


if __name__ == "__main__":
    main()
