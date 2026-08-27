#!/usr/bin/env python3
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "04_main20k_split" / "train.jsonl"
LOSS_PATH = BASE_DIR / "22_base_train_loss_profile_frozen" / "base_train_loss_profile.jsonl"
PRIOR_PATH = BASE_DIR / "21_hard_dev_profile" / "structural_error_priors.json"
HARD_DEV_PATH = BASE_DIR / "21_hard_dev_profile" / "hard_development_challenge.jsonl"

OUT_DIR = BASE_DIR / "23_megd_v1_distillation"

QUOTAS_25 = {"full": 1352, "weak": 1402, "partial": 1242}
QUOTAS_50 = {"full": 2704, "weak": 2804, "partial": 2484}

SMOOTHING_STRENGTH = 50.0
DIFFICULTY_WEIGHT = 0.65
ERROR_PRIOR_WEIGHT = 0.35
LOSS_CLIP_PERCENTILE = 0.90

AVG_PATTERNS = {
    "calories": re.compile(r"- Average calories:\s*([0-9]+(?:\.[0-9]+)?)\s*kcal", re.I),
    "protein": re.compile(r"- Average protein:\s*([0-9]+(?:\.[0-9]+)?)\s*g", re.I),
    "fiber": re.compile(r"- Average (?:fibre|fiber):\s*([0-9]+(?:\.[0-9]+)?)\s*g", re.I),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL: {path}:{line_no}: {e}") from e
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_boundary_threshold(dev_rows: list[dict[str, Any]]) -> float:
    true_margins = []
    false_margins = []

    for row in dev_rows:
        sf = row.get("structural_features", {})
        margin = sf.get("minimum_relative_margin")
        near = sf.get("near_boundary")
        if isinstance(margin, (int, float)) and isinstance(near, bool):
            (true_margins if near else false_margins).append(float(margin))

    if true_margins and false_margins:
        max_true = max(true_margins)
        min_false = min(false_margins)
        if max_true < min_false:
            return (max_true + min_false) / 2.0

    # Frozen Step-41 fallback only if the threshold cannot be inferred.
    return 0.10


def parse_average_values(input_text: str) -> dict[str, float]:
    values = {}
    for name, pattern in AVG_PATTERNS.items():
        m = pattern.search(input_text)
        if not m:
            raise RuntimeError(f"Could not parse Average {name} from training input.")
        values[name] = float(m.group(1))
    return values


def relative_margin(value: float, threshold: float) -> float:
    denom = max(abs(float(threshold)), 1e-12)
    return abs(float(value) - float(threshold)) / denom


def build_structural_bin(train_row: dict[str, Any], boundary_threshold: float) -> tuple[str, dict[str, Any]]:
    md = train_row["metadata"]
    constraints = md["constraints"]
    checks = md["constraint_checks"]
    match_type = str(md["match_type"]).lower()

    failed = [k for k in ("cuisine", "calories", "protein", "fiber") if not bool(checks[k])]
    failed_count = len(failed)

    if failed_count == 0:
        fail_bucket = "fail0"
    elif failed_count == 1:
        fail_bucket = "fail1"
    else:
        fail_bucket = "fail2plus"

    cuisine_mismatch = not bool(checks["cuisine"])

    avgs = parse_average_values(train_row["input"])
    numeric_margins = {
        "calories": relative_margin(avgs["calories"], float(constraints["max_calories"])),
        "protein": relative_margin(avgs["protein"], float(constraints["min_protein"])),
        "fiber": relative_margin(avgs["fiber"], float(constraints["min_fiber"])),
    }
    min_margin = min(numeric_margins.values())
    near_boundary = min_margin <= boundary_threshold

    key = (
        f"{match_type}|{fail_bucket}|"
        f"{'boundary' if near_boundary else 'nonboundary'}|"
        f"{'cuisine_mismatch' if cuisine_mismatch else 'cuisine_match'}"
    )

    features = {
        "match_type": match_type,
        "failed_constraints": failed,
        "failed_constraint_count": failed_count,
        "cuisine_mismatch": cuisine_mismatch,
        "minimum_relative_margin": min_margin,
        "near_boundary": near_boundary,
        "numeric_relative_margins": numeric_margins,
        "structural_bin": key,
    }
    return key, features


def build_smoothed_priors(prior_obj: dict[str, Any]) -> tuple[dict[str, float], float]:
    priors = prior_obj["structural_priors"]
    total_samples = sum(int(v["samples"]) for v in priors.values())
    total_errors = sum(int(v["high_confidence_errors"]) for v in priors.values())
    global_error_rate = total_errors / total_samples

    smoothed = {}
    for key, v in priors.items():
        n = int(v["samples"])
        e = int(v["high_confidence_errors"])
        smoothed[key] = (
            e + SMOOTHING_STRENGTH * global_error_rate
        ) / (n + SMOOTHING_STRENGTH)

    return smoothed, global_error_rate


def percentile_difficulty(loss_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_match = defaultdict(list)
    for row in loss_rows:
        by_match[str(row["match_type"]).lower()].append(float(row["base_teacher_forced_loss"]))

    sorted_losses = {m: sorted(vals) for m, vals in by_match.items()}

    result = {}
    for row in loss_rows:
        sid = row["sample_id"]
        match = str(row["match_type"]).lower()
        loss = float(row["base_teacher_forced_loss"])
        vals = sorted_losses[match]

        # Empirical CDF within match type.
        pct = bisect.bisect_right(vals, loss) / len(vals)
        difficulty = min(pct / LOSS_CLIP_PERCENTILE, 1.0)

        result[sid] = {
            "loss": loss,
            "loss_percentile_within_match": pct,
            "difficulty_score": difficulty,
        }

    return result


def median_anchor_per_signature(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sig = defaultdict(list)
    for item in items:
        by_sig[item["constraint_signature_id"]].append(item)

    anchors = []
    for sig, group in sorted(by_sig.items()):
        score_values = [x["meg_score"] for x in group]
        med = statistics.median(score_values)
        anchor = min(
            group,
            key=lambda x: (abs(x["meg_score"] - med), x["sample_id"]),
        )
        anchors.append(anchor)
    return anchors


def select_subset(
    items: list[dict[str, Any]],
    quotas: dict[str, int],
    initial_ids: set[str] | None = None,
    require_signature_anchors: bool = False,
) -> list[dict[str, Any]]:
    by_id = {x["sample_id"]: x for x in items}
    selected_ids = set(initial_ids or set())

    if require_signature_anchors:
        for anchor in median_anchor_per_signature(items):
            selected_ids.add(anchor["sample_id"])

    selected_counts = Counter(by_id[sid]["match_type"] for sid in selected_ids)

    for match_type, quota in quotas.items():
        if selected_counts[match_type] > quota:
            raise RuntimeError(
                f"Initial selection exceeds {match_type} quota: "
                f"{selected_counts[match_type]} > {quota}"
            )

        candidates = [
            x for x in items
            if x["match_type"] == match_type and x["sample_id"] not in selected_ids
        ]
        candidates.sort(key=lambda x: (-x["meg_score"], x["sample_id"]))

        need = quota - selected_counts[match_type]
        if len(candidates) < need:
            raise RuntimeError(
                f"Not enough candidates for {match_type}: need={need}, available={len(candidates)}"
            )

        for x in candidates[:need]:
            selected_ids.add(x["sample_id"])

    selected = [by_id[sid] for sid in selected_ids]
    selected.sort(key=lambda x: x["sample_id"])

    got = Counter(x["match_type"] for x in selected)
    if dict(got) != dict(Counter(quotas)):
        raise RuntimeError(f"Quota mismatch: got={dict(got)}, expected={quotas}")

    return selected


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    match_counts = Counter(x["match_type"] for x in items)
    sig_counts = Counter(x["constraint_signature_id"] for x in items)
    query_counts = Counter(x["query_id"] for x in items)
    restaurant_counts = Counter(x["restaurant_name"] for x in items)
    goal_counts = Counter(x["goal"] for x in items)
    cuisine_counts = Counter(x["cuisine"] for x in items)
    structural_counts = Counter(x["structural_bin"] for x in items)

    def dist(vals: list[float]) -> dict[str, float]:
        vals = sorted(vals)
        if not vals:
            return {}
        def q(p: float) -> float:
            idx = min(len(vals) - 1, max(0, math.ceil(p * len(vals)) - 1))
            return vals[idx]
        return {
            "min": vals[0],
            "p10": q(0.10),
            "median": statistics.median(vals),
            "p90": q(0.90),
            "max": vals[-1],
            "mean": statistics.fmean(vals),
        }

    most_rest = restaurant_counts.most_common(1)[0] if restaurant_counts else (None, 0)
    most_query = query_counts.most_common(1)[0] if query_counts else (None, 0)

    return {
        "samples": len(items),
        "match_counts": dict(sorted(match_counts.items())),
        "unique_signatures": len(sig_counts),
        "unique_queries": len(query_counts),
        "unique_restaurants": len(restaurant_counts),
        "goal_counts": dict(sorted(goal_counts.items())),
        "cuisine_counts": dict(sorted(cuisine_counts.items())),
        "structural_bin_counts": dict(sorted(structural_counts.items())),
        "meg_score": dist([x["meg_score"] for x in items]),
        "difficulty_score": dist([x["difficulty_score"] for x in items]),
        "smoothed_error_prior": dist([x["smoothed_error_prior"] for x in items]),
        "base_teacher_forced_loss": dist([x["base_teacher_forced_loss"] for x in items]),
        "max_samples_from_one_query": {
            "query_id": most_query[0],
            "count": most_query[1],
            "share": most_query[1] / len(items) if items else 0.0,
        },
        "max_samples_from_one_restaurant": {
            "restaurant_name": most_rest[0],
            "count": most_rest[1],
            "share": most_rest[1] / len(items) if items else 0.0,
        },
    }


def main() -> None:
    for p in (TRAIN_PATH, LOSS_PATH, PRIOR_PATH, HARD_DEV_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    print("=" * 72)
    print("STEP 43 — MEGD-v1 MODEL-AWARE ERROR-GUIDED DISTILLATION")
    print("=" * 72)

    train_rows = load_jsonl(TRAIN_PATH)
    loss_rows = load_jsonl(LOSS_PATH)
    dev_rows = load_jsonl(HARD_DEV_PATH)

    with PRIOR_PATH.open("r", encoding="utf-8") as f:
        prior_obj = json.load(f)

    if len(train_rows) != 15983:
        raise RuntimeError(f"Expected 15983 train rows, got {len(train_rows)}")
    if len(loss_rows) != len(train_rows):
        raise RuntimeError(
            f"Train/loss row count mismatch: train={len(train_rows)} loss={len(loss_rows)}"
        )

    train_by_id = {r["sample_id"]: r for r in train_rows}
    loss_by_id = {r["sample_id"]: r for r in loss_rows}

    if len(train_by_id) != len(train_rows):
        raise RuntimeError("Duplicate sample_id in train.jsonl")
    if len(loss_by_id) != len(loss_rows):
        raise RuntimeError("Duplicate sample_id in loss profile")
    if set(train_by_id) != set(loss_by_id):
        missing_loss = sorted(set(train_by_id) - set(loss_by_id))[:10]
        missing_train = sorted(set(loss_by_id) - set(train_by_id))[:10]
        raise RuntimeError(
            f"sample_id mismatch. Missing loss={missing_loss}, missing train={missing_train}"
        )

    boundary_threshold = infer_boundary_threshold(dev_rows)
    smoothed_priors, global_error_rate = build_smoothed_priors(prior_obj)
    difficulty = percentile_difficulty(loss_rows)

    print(f"Train samples              : {len(train_rows)}")
    print(f"Development profile rows   : {len(dev_rows)}")
    print(f"Global high-conf error rate: {global_error_rate:.6f}")
    print(f"Inferred boundary threshold: {boundary_threshold:.6f}")
    print(f"Smoothing strength         : {SMOOTHING_STRENGTH:g}")
    print(
        f"Score                      : "
        f"{DIFFICULTY_WEIGHT:.2f}*difficulty + "
        f"{ERROR_PRIOR_WEIGHT:.2f}*error_prior"
    )
    print()

    scored_items = []
    missing_prior_bins = Counter()

    for row in train_rows:
        sid = row["sample_id"]
        md = row["metadata"]

        structural_bin, sf = build_structural_bin(row, boundary_threshold)
        if structural_bin not in smoothed_priors:
            missing_prior_bins[structural_bin] += 1
            continue

        d = difficulty[sid]
        e = smoothed_priors[structural_bin]
        meg_score = DIFFICULTY_WEIGHT * d["difficulty_score"] + ERROR_PRIOR_WEIGHT * e

        scored_items.append({
            "sample_id": sid,
            "query_id": md["query_id"],
            "constraint_signature_id": md["constraint_signature_id"],
            "restaurant_name": md["restaurant_name"],
            "match_type": str(md["match_type"]).lower(),
            "goal": md["constraints"]["goal"],
            "cuisine": md["constraints"]["cuisine"],
            "base_teacher_forced_loss": d["loss"],
            "loss_percentile_within_match": d["loss_percentile_within_match"],
            "difficulty_score": d["difficulty_score"],
            "smoothed_error_prior": e,
            "meg_score": meg_score,
            **sf,
        })

    if missing_prior_bins:
        raise RuntimeError(
            "Some training structural bins were absent from frozen Dev priors: "
            + json.dumps(dict(missing_prior_bins), ensure_ascii=False, sort_keys=True)
        )

    if len(scored_items) != len(train_rows):
        raise RuntimeError(
            f"Scored sample mismatch: {len(scored_items)} vs {len(train_rows)}"
        )

    # 25%: one representative median-score anchor per signature, then model-aware fill.
    selected_25 = select_subset(
        scored_items,
        quotas=QUOTAS_25,
        initial_ids=None,
        require_signature_anchors=True,
    )
    ids_25 = {x["sample_id"] for x in selected_25}

    # 50%: nested extension of the frozen 25% subset.
    selected_50 = select_subset(
        scored_items,
        quotas=QUOTAS_50,
        initial_ids=ids_25,
        require_signature_anchors=False,
    )
    ids_50 = {x["sample_id"] for x in selected_50}

    if not ids_25.issubset(ids_50):
        raise RuntimeError("Nestedness failed: 25% is not a subset of 50%")

    if len(selected_25) != 3996:
        raise RuntimeError(f"25% count mismatch: {len(selected_25)}")
    if len(selected_50) != 7992:
        raise RuntimeError(f"50% count mismatch: {len(selected_50)}")

    sig25 = {x["constraint_signature_id"] for x in selected_25}
    sig50 = {x["constraint_signature_id"] for x in selected_50}

    if len(sig25) != 500 or len(sig50) != 500:
        raise RuntimeError(
            f"Signature coverage failed: 25%={len(sig25)}, 50%={len(sig50)}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve original training row schema for trainer compatibility.
    train_25 = [train_by_id[x["sample_id"]] for x in selected_25]
    train_50 = [train_by_id[x["sample_id"]] for x in selected_50]
    train_25.sort(key=lambda r: r["sample_id"])
    train_50.sort(key=lambda r: r["sample_id"])

    path25 = OUT_DIR / "train_megd_25pct.jsonl"
    path50 = OUT_DIR / "train_megd_50pct.jsonl"
    man25 = OUT_DIR / "selection_manifest_25pct.jsonl"
    man50 = OUT_DIR / "selection_manifest_50pct.jsonl"
    summary_path = OUT_DIR / "megd_v1_summary.json"

    dump_jsonl(path25, train_25)
    dump_jsonl(path50, train_50)
    dump_jsonl(man25, selected_25)
    dump_jsonl(man50, selected_50)

    summary = {
        "method": "MEGD-v1",
        "status": "development_method_selection_only_not_blind_test",
        "definition": (
            "Model-aware error-guided training-data distillation using "
            "within-match base-model teacher-forced-loss percentile plus "
            "empirical-Bayes-smoothed high-confidence Development Benchmark "
            "structural error prior, with representative signature anchors "
            "and frozen match-type quotas."
        ),
        "inputs": {
            "train": str(TRAIN_PATH),
            "loss_profile": str(LOSS_PATH),
            "structural_error_priors": str(PRIOR_PATH),
            "hard_development_profile": str(HARD_DEV_PATH),
            "train_sha256": sha256(TRAIN_PATH),
            "loss_profile_sha256": sha256(LOSS_PATH),
            "structural_error_priors_sha256": sha256(PRIOR_PATH),
            "hard_development_profile_sha256": sha256(HARD_DEV_PATH),
        },
        "parameters": {
            "difficulty_weight": DIFFICULTY_WEIGHT,
            "error_prior_weight": ERROR_PRIOR_WEIGHT,
            "loss_percentile_scope": "within_match_type",
            "loss_clip_percentile": LOSS_CLIP_PERCENTILE,
            "empirical_bayes_smoothing_strength": SMOOTHING_STRENGTH,
            "global_dev_high_confidence_error_rate": global_error_rate,
            "inferred_near_boundary_threshold": boundary_threshold,
            "anchor_policy": "one median-MEG-score representative sample per constraint signature in 25pct",
            "nested_policy": "25pct subset is a strict subset of 50pct",
            "quota_25pct": QUOTAS_25,
            "quota_50pct": QUOTAS_50,
        },
        "full_train": summarize(scored_items),
        "meg_25pct": summarize(selected_25),
        "meg_50pct": summarize(selected_50),
        "checks": {
            "samples_25pct": len(selected_25),
            "samples_50pct": len(selected_50),
            "unique_signatures_25pct": len(sig25),
            "unique_signatures_50pct": len(sig50),
            "nested_25_in_50": ids_25.issubset(ids_50),
            "blind_test_used": False,
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    hashes = {
        path25.name: sha256(path25),
        path50.name: sha256(path50),
        man25.name: sha256(man25),
        man50.name: sha256(man50),
        summary_path.name: sha256(summary_path),
    }
    with (OUT_DIR / "sha256sums.txt").open("w", encoding="utf-8") as f:
        for name, digest in hashes.items():
            f.write(f"{digest}  {name}\n")

    print("=" * 72)
    print("MEGD-v1 SELECTION COMPLETE")
    print("=" * 72)
    for label, selected in (("25%", selected_25), ("50%", selected_50)):
        s = summarize(selected)
        print(
            f"{label:>3}  samples={s['samples']}  "
            f"signatures={s['unique_signatures']}  "
            f"queries={s['unique_queries']}  "
            f"restaurants={s['unique_restaurants']}  "
            f"match={s['match_counts']}  "
            f"mean_score={s['meg_score']['mean']:.4f}"
        )

    print(f"Nested 25% ⊂ 50% : {ids_25.issubset(ids_50)}")
    print(f"Blind test used   : False")
    print(f"Output            : {OUT_DIR}")
    print()
    print("SHA256:")
    for name, digest in hashes.items():
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
