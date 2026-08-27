#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

TRAIN_POOL = HERE / "04_main20k_split" / "train.jsonl"
BASE_TRAINER = HERE / "step35_train_main20k_matched_random_50pct.py"

EVAL_CANDIDATES = [
    HERE / "step36_evaluate_matched_random_50pct.py",
    HERE / "step45_evaluate_megd_50pct_dev.py",
    HERE / "step31_evaluate_bcaegd_50pct.py",
]

OUT_ROOT = HERE / "28_low_budget_random_multiseed"
SCRIPT_ROOT = HERE

SEEDS = (20260813, 20260814, 20260815)
BUDGETS = {
    "10pct": 1598,
    "5pct": 799,
}

FULL_COUNTS = {
    "full": 5408,
    "weak": 5608,
    "partial": 4967,
}
MATCH_ORDER = ("full", "weak", "partial")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def md(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def largest_remainder_targets(target_n: int) -> dict[str, int]:
    total = sum(FULL_COUNTS.values())
    raw = {k: target_n * v / total for k, v in FULL_COUNTS.items()}
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


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
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
            str(md(r).get("match_type", ""))
            for r in rows
        )),
    }


def choose_signature_anchors(
    by_sig: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """
    Select one random sample per signature while staying inside final
    Full/Weak/Partial quotas. Multiple deterministic attempts avoid greedy
    dead ends at the very small 5% budget.
    """
    signatures = sorted(by_sig)

    for attempt in range(500):
        rng = random.Random(seed + attempt * 100003)
        current = Counter()
        chosen: list[dict[str, Any]] = []

        # Process signatures with fewer available match types first.
        decorated = []
        for sid in signatures:
            available = {
                str(md(r).get("match_type", ""))
                for r in by_sig[sid]
            }
            decorated.append((len(available), rng.random(), sid))
        decorated.sort()

        failed = False
        for _, _, sid in decorated:
            candidates = [
                r for r in by_sig[sid]
                if current[str(md(r).get("match_type", ""))]
                < quotas[str(md(r).get("match_type", ""))]
            ]
            if not candidates:
                failed = True
                break

            # Prefer the match type with the largest remaining capacity.
            remaining_by_match = {
                match: quotas[match] - current[match]
                for match in MATCH_ORDER
            }
            max_remaining = max(
                remaining_by_match[str(md(r).get("match_type", ""))]
                for r in candidates
            )
            candidates = [
                r for r in candidates
                if remaining_by_match[
                    str(md(r).get("match_type", ""))
                ] == max_remaining
            ]
            chosen_row = rng.choice(candidates)
            chosen.append(chosen_row)
            current[str(md(chosen_row).get("match_type", ""))] += 1

        if not failed:
            return chosen

    raise RuntimeError(
        "Could not construct all-signature anchors within match quotas "
        "after 500 deterministic attempts."
    )


def fill_to_quota(
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result = list(selected)
    selected_ids = {str(r["sample_id"]) for r in result}
    current = Counter(str(md(r).get("match_type", "")) for r in result)

    for match in MATCH_ORDER:
        need = quotas[match] - current[match]
        if need < 0:
            raise RuntimeError(
                f"Anchor selection exceeded {match} quota: "
                f"{current[match]} > {quotas[match]}"
            )

        candidates = [
            r for r in pool
            if str(r["sample_id"]) not in selected_ids
            and str(md(r).get("match_type", "")) == match
        ]
        if len(candidates) < need:
            raise RuntimeError(
                f"Not enough {match} candidates: need={need}, "
                f"available={len(candidates)}"
            )

        for row in rng.sample(candidates, need):
            result.append(row)
            selected_ids.add(str(row["sample_id"]))

    result.sort(key=lambda r: str(r["sample_id"]))
    return result


def extend_nested(
    pool: list[dict[str, Any]],
    smaller: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    return fill_to_quota(
        pool=pool,
        selected=smaller,
        quotas=quotas,
        seed=seed,
    )


def patch_training_script(
    source: str,
    subset_rel: str,
    run_dir: str,
    train_count: int,
    experiment_name: str,
) -> str:
    text = source

    text, n = re.subn(
        r'^TRAIN_FILE\s*=.*$',
        f'TRAIN_FILE = EXPERIMENT_ROOT / "{subset_rel.split("/")[0]}"'
        + "".join(f' / "{part}"' for part in subset_rel.split("/")[1:]),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("Could not patch TRAIN_FILE")

    text, n = re.subn(
        r'^RUN_ROOT\s*=.*$',
        f'RUN_ROOT = EXPERIMENT_ROOT / "{run_dir}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("Could not patch RUN_ROOT")

    text, n = re.subn(
        r'^EXPECTED_TRAIN_SAMPLES\s*=\s*\d+\s*$',
        f'EXPECTED_TRAIN_SAMPLES = {train_count}',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("Could not patch EXPECTED_TRAIN_SAMPLES")

    total = train_count + 1948 + 2069
    text, n = re.subn(
        r'^EXPECTED_TOTAL_SAMPLES\s*=\s*\d+\s*$',
        f'EXPECTED_TOTAL_SAMPLES = {total}',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("Could not patch EXPECTED_TOTAL_SAMPLES")

    text, n = re.subn(
        r'"experiment"\s*:\s*"[^"]+"',
        f'"experiment": "{experiment_name}"',
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not patch experiment name")

    return text


def patch_eval_script(
    source: str,
    adapter_run_dir: str,
    eval_root: str,
) -> str:
    text = source

    adapter_candidates = [
        "15_main20k_qlora_matched_random_50pct",
        "24_main20k_qlora_megd_50pct",
        "10_main20k_qlora_bcaegd_50pct",
    ]
    replaced_adapter = False
    for old in adapter_candidates:
        if old in text:
            text = text.replace(old, adapter_run_dir)
            replaced_adapter = True
            break
    if not replaced_adapter:
        raise RuntimeError(
            "Could not identify adapter run directory in evaluation source."
        )

    eval_candidates = [
        "16_matched_random_50pct_eval",
        "26_megd_50pct_dev_eval",
        "11_bcaegd_50pct_eval",
    ]
    replaced_eval = False
    for old in eval_candidates:
        if old in text:
            text = text.replace(old, eval_root)
            replaced_eval = True
            break
    if not replaced_eval:
        raise RuntimeError(
            "Could not identify output root in evaluation source."
        )

    # Main-20K accepted records use the frozen Filter-v2.3 field.
    text = text.replace(
        'metadata_of(record).get("accepted")',
        'metadata_of(record).get("filter_v2_3_accepted")',
    )

    return text


def run(cmd: list[str], env: dict[str, str]) -> None:
    print()
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def main() -> None:
    if not TRAIN_POOL.exists():
        raise FileNotFoundError(TRAIN_POOL)
    if not BASE_TRAINER.exists():
        raise FileNotFoundError(BASE_TRAINER)

    eval_source_path = next(
        (p for p in EVAL_CANDIDATES if p.exists()),
        None,
    )
    if eval_source_path is None:
        raise FileNotFoundError(
            "No validated Main-20K evaluation script was found."
        )

    rows = read_jsonl(TRAIN_POOL)
    if len(rows) != 15983:
        raise RuntimeError(f"Expected 15983 train rows, got {len(rows)}")

    by_sig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = str(md(row).get("constraint_signature_id", ""))
        if not sid:
            raise RuntimeError("Missing constraint_signature_id")
        by_sig[sid].append(row)

    if len(by_sig) != 500:
        raise RuntimeError(f"Expected 500 signatures, got {len(by_sig)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)

    trainer_source = BASE_TRAINER.read_text(encoding="utf-8")
    eval_source = eval_source_path.read_text(encoding="utf-8")

    prepared: list[dict[str, Any]] = []
    master_summary: dict[str, Any] = {
        "experiment": "low_budget_matched_random_multiseed",
        "development_only": True,
        "blind_test_used": False,
        "seeds": list(SEEDS),
        "budgets": {},
    }

    print("=" * 72)
    print("LOW-BUDGET MATCHED RANDOM MULTI-SEED PIPELINE")
    print("=" * 72)
    print("Train pool       :", len(rows))
    print("Signatures       :", len(by_sig))
    print("Seeds            :", SEEDS)
    print("Budgets          :", BUDGETS)
    print("Evaluation source:", eval_source_path.name)

    for seed in SEEDS:
        seed_dir = OUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        quotas5 = largest_remainder_targets(BUDGETS["5pct"])
        anchors5 = choose_signature_anchors(
            by_sig=by_sig,
            quotas=quotas5,
            seed=seed,
        )
        subset5 = fill_to_quota(
            pool=rows,
            selected=anchors5,
            quotas=quotas5,
            seed=seed + 5,
        )

        quotas10 = largest_remainder_targets(BUDGETS["10pct"])
        subset10 = extend_nested(
            pool=rows,
            smaller=subset5,
            quotas=quotas10,
            seed=seed + 10,
        )

        ids5 = {str(r["sample_id"]) for r in subset5}
        ids10 = {str(r["sample_id"]) for r in subset10}
        if not ids5 <= ids10:
            raise RuntimeError(f"Nestedness failed for seed {seed}")

        if len(subset5) != 799 or len(subset10) != 1598:
            raise RuntimeError(f"Subset count mismatch for seed {seed}")

        for budget, subset, quotas in (
            ("5pct", subset5, quotas5),
            ("10pct", subset10, quotas10),
        ):
            subset_name = f"train_random_{budget}.jsonl"
            subset_path = seed_dir / subset_name
            write_jsonl(subset_path, subset)

            cov = coverage(subset)
            if cov["signatures"] != 500:
                raise RuntimeError(
                    f"{budget} seed {seed} lost signature coverage."
                )
            if cov["match"] != quotas:
                raise RuntimeError(
                    f"{budget} seed {seed} match mismatch: "
                    f"{cov['match']} vs {quotas}"
                )

            train_count = len(subset)
            run_dir = f"29_random_{budget}_seed_{seed}"
            eval_root = f"30_random_{budget}_seed_{seed}_dev_eval"
            experiment_name = (
                f"tinyllama_qlora_main20k_random_{budget}_seed_{seed}"
            )

            train_script = (
                SCRIPT_ROOT
                / f"train_random_{budget}_seed_{seed}.py"
            )
            eval_script = (
                SCRIPT_ROOT
                / f"eval_random_{budget}_seed_{seed}.py"
            )

            subset_rel = str(subset_path.relative_to(HERE))
            train_script.write_text(
                patch_training_script(
                    source=trainer_source,
                    subset_rel=subset_rel,
                    run_dir=run_dir,
                    train_count=train_count,
                    experiment_name=experiment_name,
                ),
                encoding="utf-8",
            )
            eval_script.write_text(
                patch_eval_script(
                    source=eval_source,
                    adapter_run_dir=run_dir,
                    eval_root=eval_root,
                ),
                encoding="utf-8",
            )

            prepared.append({
                "seed": seed,
                "budget": budget,
                "train_script": train_script,
                "eval_script": eval_script,
                "run_dir": HERE / run_dir,
                "eval_dir": HERE / eval_root,
                "coverage": cov,
            })

            master_summary["budgets"][
                f"{budget}_seed_{seed}"
            ] = {
                **cov,
                "quotas": quotas,
                "subset_file": str(subset_path.relative_to(HERE)),
                "nested_5_in_10": ids5 <= ids10,
            }

    (OUT_ROOT / "subset_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Compile all generated scripts before any long GPU work.
    for job in prepared:
        run(
            [sys.executable, "-m", "py_compile", str(job["train_script"])],
            env=os.environ.copy(),
        )
        run(
            [sys.executable, "-m", "py_compile", str(job["eval_script"])],
            env=os.environ.copy(),
        )

    env = os.environ.copy()
    env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # Run 10% first, then 5%, each across all seeds.
    prepared.sort(
        key=lambda x: (
            0 if x["budget"] == "10pct" else 1,
            x["seed"],
        )
    )

    for index, job in enumerate(prepared, 1):
        seed = job["seed"]
        budget = job["budget"]
        run_dir: Path = job["run_dir"]
        eval_dir: Path = job["eval_dir"]
        final_adapter = run_dir / "full_run" / "final_adapter"
        frozen_dir = run_dir / "full_run_frozen"
        eval_summary = eval_dir / "test_final_2069" / "evaluation_summary.json"

        print()
        print("=" * 72)
        print(
            f"JOB {index}/{len(prepared)} — "
            f"RANDOM {budget} SEED {seed}"
        )
        print("=" * 72)

        if not final_adapter.exists():
            run(
                [
                    sys.executable,
                    str(job["train_script"]),
                    "--smoke-test",
                    "--overwrite",
                ],
                env=env,
            )
            run(
                [
                    sys.executable,
                    str(job["train_script"]),
                    "--overwrite",
                ],
                env=env,
            )
        else:
            print("Training already complete; skipping.")

        if not frozen_dir.exists():
            shutil.copytree(run_dir / "full_run", frozen_dir)
        else:
            print("Frozen model already exists; skipping copy.")

        if not eval_summary.exists():
            run(
                [
                    sys.executable,
                    str(job["eval_script"]),
                    "--split",
                    "test",
                    "--batch-size",
                    "8",
                    "--overwrite",
                ],
                env=env,
            )
        else:
            print("Evaluation already complete; skipping.")

    print()
    print("=" * 72)
    print("LOW-BUDGET MULTI-SEED PIPELINE COMPLETE")
    print("=" * 72)
    print("Blind test used: False")
    print("Subset summary:", OUT_ROOT / "subset_summary.json")


if __name__ == "__main__":
    main()
