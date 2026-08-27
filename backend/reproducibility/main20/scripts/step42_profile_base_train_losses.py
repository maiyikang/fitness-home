from __future__ import annotations

import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
TRAIN_FILE = HERE / "04_main20k_split" / "train.jsonl"
TRAINER_FILE = HERE / "step24_train_main20k_100pct.py"

OUT_DIR = HERE / "22_base_train_loss_profile"
PROFILE_FILE = OUT_DIR / "base_train_loss_profile.jsonl"
SUMMARY_FILE = OUT_DIR / "base_train_loss_summary.json"

BATCH_SIZE = 8


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def quantiles(values):
    if not values:
        return {}
    x = sorted(values)

    def q(p):
        pos = (len(x) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return x[lo]
        frac = pos - lo
        return x[lo] * (1 - frac) + x[hi] * frac

    return {
        "min": x[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": x[-1],
        "mean": sum(x) / len(x),
    }


def main():
    for path in (TRAIN_FILE, TRAINER_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    trainer = load_module(TRAINER_FILE, "fh_main20k_trainer")
    records = read_jsonl(TRAIN_FILE)

    if len(records) != 15983:
        raise RuntimeError(f"Expected 15983 train samples, got {len(records)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = trainer.load_tokenizer()
    dataset = trainer.FitnessHomeCausalDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=trainer.MAX_LENGTH,
        split_name="train_loss_profile",
    )
    collator = trainer.CausalLMCollator(
        pad_token_id=tokenizer.pad_token_id,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("=" * 72)
    print("BASE MODEL TRAIN-POOL LOSS PROFILING")
    print("=" * 72)
    print("Samples    :", len(records))
    print("Batch size :", BATCH_SIZE)
    print("Model      :", trainer.BASE_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        trainer.BASE_MODEL_NAME,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.config.use_cache = False

    rows = []
    all_losses = []
    by_match = defaultdict(list)
    by_failed_count = defaultdict(list)

    start = time.time()
    offset = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, 1):
            batch = {
                key: value.to(model.device)
                for key, value in batch.items()
            }

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits[:, :-1, :].float()
            labels = batch["labels"][:, 1:]

            valid = labels.ne(-100)
            safe_labels = labels.masked_fill(~valid, 0)

            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                safe_labels.reshape(-1),
                reduction="none",
            ).view_as(labels)

            token_loss = token_loss * valid
            token_counts = valid.sum(dim=1).clamp_min(1)
            sample_losses = token_loss.sum(dim=1) / token_counts

            batch_size = sample_losses.size(0)

            for j in range(batch_size):
                record = records[offset + j]
                md = record.get("metadata", {}) or {}
                checks = md.get("constraint_checks", {}) or {}

                failed_count = sum(
                    value is False
                    for value in checks.values()
                )
                match = str(
                    md.get("match_type", "unknown")
                )
                loss = float(
                    sample_losses[j].item()
                )

                row = {
                    "sample_id": record.get("sample_id"),
                    "query_id": md.get("query_id"),
                    "constraint_signature_id": md.get(
                        "constraint_signature_id"
                    ),
                    "restaurant_name": md.get(
                        "restaurant_name"
                    ),
                    "match_type": match,
                    "constraints": md.get("constraints"),
                    "constraint_checks": checks,
                    "failed_constraint_count": failed_count,
                    "target_token_count": int(
                        token_counts[j].item()
                    ),
                    "base_teacher_forced_loss": loss,
                }

                rows.append(row)
                all_losses.append(loss)
                by_match[match].append(loss)
                by_failed_count[
                    str(failed_count)
                ].append(loss)

            offset += batch_size

            if batch_idx % 100 == 0 or offset == len(records):
                elapsed = time.time() - start
                rate = offset / elapsed if elapsed else 0.0
                print(
                    f"[{offset:05d}/{len(records)}] "
                    f"{rate:.1f} samples/s"
                )

    if len(rows) != len(records):
        raise RuntimeError(
            f"Profile count mismatch: {len(rows)}"
        )

    write_jsonl(PROFILE_FILE, rows)

    ranked = sorted(
        rows,
        key=lambda r: r["base_teacher_forced_loss"],
        reverse=True,
    )

    summary = {
        "profile_version": "base_train_teacher_forced_loss_v1",
        "samples": len(rows),
        "batch_size": BATCH_SIZE,
        "model": trainer.BASE_MODEL_NAME,
        "loss_distribution": quantiles(all_losses),
        "loss_by_match_type": {
            key: quantiles(values)
            for key, values in sorted(by_match.items())
        },
        "loss_by_failed_constraint_count": {
            key: quantiles(values)
            for key, values in sorted(by_failed_count.items())
        },
        "highest_loss_examples": [
            {
                "sample_id": row["sample_id"],
                "match_type": row["match_type"],
                "failed_constraint_count": row[
                    "failed_constraint_count"
                ],
                "loss": row[
                    "base_teacher_forced_loss"
                ],
            }
            for row in ranked[:20]
        ],
        "important_note": (
            "This score uses only the Main-20K training pool and "
            "the base TinyLlama model. The reserved final-blind "
            "signatures are not accessed."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BASE TRAIN LOSS PROFILE COMPLETE")
    print("=" * 72)
    print("Samples :", len(rows))
    print(
        "Loss distribution:",
        json.dumps(
            summary["loss_distribution"],
            ensure_ascii=False,
        ),
    )
    print("Loss by match:")
    for key, value in summary[
        "loss_by_match_type"
    ].items():
        print(
            f"  {key:<8} mean={value['mean']:.4f} "
            f"median={value['median']:.4f} "
            f"p90={value['p90']:.4f}"
        )
    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
