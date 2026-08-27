from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =====================================================================
# Frozen experiment configuration
# =====================================================================

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEED = 42

MAX_LENGTH = 512
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 0.3

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

SYSTEM_PROMPT = """
You are the Fitness Home recommendation explanation model.
The retrieval system has already selected the restaurant.
Use only the supplied evidence and constraint evaluation.
Do not recommend a different restaurant, invent facts, alter numbers,
or infer unsupported health benefits. Clearly state unmet constraints.
Return one concise evidence-grounded paragraph only.
""".strip()


# =====================================================================
# Paths
# =====================================================================

SCRIPT_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_FILE.parent
PROJECT_ROOT = SCRIPT_FILE.parents[3]

DATASET_DIR = EXPERIMENT_ROOT / "04_main20k_split"
TRAIN_FILE = EXPERIMENT_ROOT / "34_signature_coverage_ablation" / "seed_20260814" / "train_no_signature_10pct.jsonl"
VALIDATION_FILE = DATASET_DIR / "validation.jsonl"
TEST_FILE = DATASET_DIR / "test.jsonl"

RUN_ROOT = EXPERIMENT_ROOT / "35_no_signature_10pct_seed_20260814"
FULL_RUN_DIR = RUN_ROOT / "full_run"
SMOKE_RUN_DIR = RUN_ROOT / "smoke_test"

EXPECTED_TRAIN_SAMPLES = 1598
EXPECTED_VALIDATION_SAMPLES = 1948
EXPECTED_TEST_SAMPLES = 2069
EXPECTED_TOTAL_SAMPLES = 5615
# =====================================================================
# Utility functions
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TinyLlama with QLoRA on the frozen Fitness Home "
            "retrieval-grounded explanation dataset."
        )
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a 3-step integration test on a small subset. "
            "This writes only to the smoke_test directory."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint in the selected run directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete the selected run directory before starting. "
            "Do not combine this with --resume."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def validate_source_records(
    train_records: Sequence[Dict[str, Any]],
    validation_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
) -> None:
    expected_counts = {
        "train": (len(train_records), EXPECTED_TRAIN_SAMPLES),
        "validation": (
            len(validation_records),
            EXPECTED_VALIDATION_SAMPLES,
        ),
        "test": (len(test_records), EXPECTED_TEST_SAMPLES),
    }

    for split_name, (actual, expected) in expected_counts.items():
        if actual != expected:
            raise RuntimeError(
                f"Unexpected {split_name} sample count: "
                f"expected {expected}, found {actual}."
            )

    all_records = [
        *train_records,
        *validation_records,
        *test_records,
    ]

    if len(all_records) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SAMPLES} total samples, "
            f"found {len(all_records)}."
        )

    sample_ids: List[str] = []
    split_queries: Dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    split_signatures: Dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    for split_name, records in (
        ("train", train_records),
        ("validation", validation_records),
        ("test", test_records),
    ):
        for record in records:
            sample_id = str(record.get("sample_id", "")).strip()
            metadata = record.get("metadata") or {}
            query_id = str(metadata.get("query_id", "")).strip()
            signature_id = str(
                metadata.get("constraint_signature_id", "")
            ).strip()

            if not sample_id:
                raise RuntimeError(f"Missing sample_id in {split_name} split.")

            if not query_id:
                raise RuntimeError(
                    f"Missing query_id in sample {sample_id}."
                )

            if not signature_id:
                raise RuntimeError(
                    "Missing constraint_signature_id in sample "
                    f"{sample_id}."
                )

            if metadata.get("filter_v2_3_accepted") is not True:
                raise RuntimeError(
                    "Sample did not pass frozen Filter v2.3: "
                    f"{sample_id}."
                )

            for field in ("instruction", "input", "output"):
                if not str(record.get(field, "")).strip():
                    raise RuntimeError(
                        f"Sample {sample_id} has an empty {field} field."
                    )

            sample_ids.append(sample_id)
            split_queries[split_name].add(query_id)
            split_signatures[split_name].add(signature_id)

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample IDs were found across splits.")

    if (
        split_queries["train"] & split_queries["validation"]
        or split_queries["train"] & split_queries["test"]
        or split_queries["validation"] & split_queries["test"]
    ):
        raise RuntimeError("Query leakage detected across dataset splits.")

    if (
        split_signatures["train"] & split_signatures["validation"]
        or split_signatures["train"] & split_signatures["test"]
        or split_signatures["validation"] & split_signatures["test"]
    ):
        raise RuntimeError(
            "Constraint-signature leakage detected across dataset splits."
        )


def newest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    candidates: List[tuple[int, Path]] = []

    if not checkpoint_dir.exists():
        return None

    for path in checkpoint_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue

        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue

        candidates.append((step, path))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }

    for package_name in (
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
    ):
        try:
            from importlib.metadata import version

            versions[package_name] = version(package_name)
        except Exception:
            versions[package_name] = "unavailable"

    return versions


# =====================================================================
# Dataset and collation
# =====================================================================


def build_user_content(record: Dict[str, Any]) -> str:
    instruction = str(record["instruction"]).strip()
    evidence_input = str(record["input"]).strip()

    redundant_marker = (
        "\n\nWrite one evidence-grounded "
        "recommendation explanation."
    )

    if redundant_marker in evidence_input:
        evidence_input = evidence_input.split(
            redundant_marker,
            1,
        )[0].strip()

    return f"{instruction}\n\n{evidence_input}"


def longest_common_prefix(
    first: Sequence[int],
    second: Sequence[int],
) -> int:
    limit = min(len(first), len(second))
    index = 0

    while index < limit and first[index] == second[index]:
        index += 1

    return index


class FitnessHomeCausalDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        split_name: str,
    ) -> None:
        self.examples: List[Dict[str, List[int]]] = []
        self.truncated_examples = 0
        self.empty_target_examples = 0
        self.max_untruncated_length = 0
        self.split_name = split_name

        for record in records:
            user_content = build_user_content(record)
            assistant_output = str(record["output"]).strip()

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # Transformers v5 returns BatchEncoding from tokenized
            # apply_chat_template calls. Render the chat template as text
            # first, then encode that text explicitly. This guarantees that
            # prompt_part is a flat list of integer token IDs across v4/v5.
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            if not isinstance(prompt_text, str) or not prompt_text.strip():
                sample_id = record.get("sample_id", "unknown")
                raise RuntimeError(
                    "Chat template returned no prompt text "
                    f"for sample {sample_id}."
                )

            prompt_part = tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )
            prompt_part = [
                int(token_id)
                for token_id in prompt_part
            ]

            if not prompt_part:
                sample_id = record.get("sample_id", "unknown")
                raise RuntimeError(
                    "Prompt tokenization returned no token IDs "
                    f"for sample {sample_id}."
                )

            target_part = tokenizer.encode(
                assistant_output,
                add_special_tokens=False,
            )
            target_part = list(target_part)

            if tokenizer.eos_token_id is not None:
                if (
                    not target_part
                    or target_part[-1] != tokenizer.eos_token_id
                ):
                    target_part.append(tokenizer.eos_token_id)

            full_ids = prompt_part + target_part

            self.max_untruncated_length = max(
                self.max_untruncated_length,
                len(full_ids),
            )

            if not target_part:
                self.empty_target_examples += 1
                sample_id = record.get("sample_id", "unknown")
                raise RuntimeError(
                    "No assistant target tokens were found "
                    f"for sample {sample_id}."
                )

            if len(full_ids) > max_length:
                self.truncated_examples += 1
                prompt_budget = max_length - len(target_part)

                if prompt_budget < 8:
                    self.empty_target_examples += 1
                    sample_id = record.get("sample_id", "unknown")
                    raise RuntimeError(
                        "Assistant target is too long to fit within "
                        f"max_length={max_length} for sample {sample_id}. "
                        f"Target tokens: {len(target_part)}."
                    )

                if len(prompt_part) > prompt_budget:
                    head_budget = min(
                        128,
                        max(4, prompt_budget // 3),
                    )
                    tail_budget = prompt_budget - head_budget

                    prompt_part = (
                        prompt_part[:head_budget]
                        + prompt_part[-tail_budget:]
                    )

                full_ids = prompt_part + target_part

            labels = (
                [-100] * len(prompt_part)
                + target_part
            )
            attention_mask = [1] * len(full_ids)

            if not any(label != -100 for label in labels):
                self.empty_target_examples += 1
                sample_id = record.get("sample_id", "unknown")
                raise RuntimeError(
                    "No assistant target tokens remain "
                    f"for sample {sample_id}."
                )

            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.examples[index]


@dataclass
class CausalLMCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(
        self,
        features: Sequence[Dict[str, List[int]]],
    ) -> Dict[str, torch.Tensor]:
        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        if self.pad_to_multiple_of > 1:
            max_length = int(
                math.ceil(
                    max_length / self.pad_to_multiple_of
                )
                * self.pad_to_multiple_of
            )

        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        labels: List[List[int]] = []

        for feature in features:
            padding_length = max_length - len(feature["input_ids"])

            input_ids.append(
                feature["input_ids"]
                + [self.pad_token_id] * padding_length
            )
            attention_masks.append(
                feature["attention_mask"]
                + [0] * padding_length
            )
            labels.append(
                feature["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


# =====================================================================
# Model and Trainer setup
# =====================================================================


def load_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    return tokenizer


def load_qlora_model() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA experiment.")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def build_training_arguments(
    output_dir: Path,
    smoke_test: bool,
) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "tf32": True,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "prediction_loss_only": True,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    parameter_names = set(signature.parameters)

    if "eval_strategy" in parameter_names:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameter_names:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "This Transformers version exposes neither eval_strategy "
            "nor evaluation_strategy in TrainingArguments."
        )

    if smoke_test:
        kwargs.update(
            {
                "max_steps": 3,
                "logging_steps": 1,
                "save_strategy": "steps",
                "save_steps": 3,
                "eval_steps": 3,
                "save_total_limit": 1,
                "load_best_model_at_end": False,
            }
        )

        if "eval_strategy" in kwargs:
            kwargs["eval_strategy"] = "steps"
        if "evaluation_strategy" in kwargs:
            kwargs["evaluation_strategy"] = "steps"

    unsupported_fields = sorted(
        key for key in kwargs
        if key not in parameter_names
    )

    for field_name in unsupported_fields:
        kwargs.pop(field_name)

    if unsupported_fields:
        print(
            "Ignoring unsupported TrainingArguments fields:",
            ", ".join(unsupported_fields),
        )

    return TrainingArguments(**kwargs)


def build_trainer(
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    collator: CausalLMCollator,
) -> Trainer:
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
    }

    trainer_signature = inspect.signature(Trainer.__init__)

    if "processing_class" in trainer_signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


# =====================================================================
# Metadata and output
# =====================================================================


def dataset_audit(
    train_dataset: FitnessHomeCausalDataset,
    validation_dataset: FitnessHomeCausalDataset,
) -> Dict[str, Any]:
    return {
        "max_length": MAX_LENGTH,
        "train": {
            "samples": len(train_dataset),
            "truncated_examples": train_dataset.truncated_examples,
            "empty_target_examples": train_dataset.empty_target_examples,
            "max_untruncated_length": (
                train_dataset.max_untruncated_length
            ),
        },
        "validation": {
            "samples": len(validation_dataset),
            "truncated_examples": (
                validation_dataset.truncated_examples
            ),
            "empty_target_examples": (
                validation_dataset.empty_target_examples
            ),
            "max_untruncated_length": (
                validation_dataset.max_untruncated_length
            ),
        },
    }


def save_run_metadata(
    path: Path,
    smoke_test: bool,
    run_dir: Path,
    train_metrics: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    token_audit: Dict[str, Any],
    elapsed_seconds: float,
    adapter_dir: Path,
) -> None:
    adapter_files = sorted(
        file
        for file in adapter_dir.rglob("*")
        if file.is_file()
    )

    metadata = {
        "experiment": "tinyllama_qlora_main20k_no_signature_random_10pct_seed_20260814",
        "smoke_test": smoke_test,
        "base_model": BASE_MODEL_NAME,
        "seed": SEED,
        "dataset": {
            "directory": str(DATASET_DIR),
            "train_file": str(TRAIN_FILE),
            "validation_file": str(VALIDATION_FILE),
            "test_file_reserved_for_final_evaluation": str(TEST_FILE),
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VALIDATION_FILE),
            "test_sha256": sha256_file(TEST_FILE),
        },
        "training_configuration": {
            "max_length": MAX_LENGTH,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": (
                PER_DEVICE_TRAIN_BATCH_SIZE
            ),
            "per_device_eval_batch_size": (
                PER_DEVICE_EVAL_BATCH_SIZE
            ),
            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_train_batch_size": (
                PER_DEVICE_TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),
            "warmup_steps": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "double_quantization": True,
                "compute_dtype": (
                    "bfloat16"
                    if torch.cuda.is_bf16_supported()
                    else "float16"
                ),
            },
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        },
        "tokenization_audit": token_audit,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "run_directory": str(run_dir),
        "adapter_directory": str(adapter_dir),
        "adapter_files": {
            str(file.relative_to(adapter_dir)): sha256_file(file)
            for file in adapter_files
        },
        "environment": {
            **package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Do not combine --resume and --overwrite.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = SMOKE_RUN_DIR if args.smoke_test else FULL_RUN_DIR
    checkpoint_dir = run_dir / "checkpoints"
    adapter_dir = run_dir / "final_adapter"
    metadata_file = run_dir / "training_metadata.json"

    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for required_file in (
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required frozen dataset file not found: {required_file}"
            )

    train_records = read_jsonl(TRAIN_FILE)
    validation_records = read_jsonl(VALIDATION_FILE)
    test_records = read_jsonl(TEST_FILE)

    validate_source_records(
        train_records,
        validation_records,
        test_records,
    )

    if args.smoke_test:
        train_records = train_records[:16]
        validation_records = validation_records[:8]

    print("=" * 76)
    print("Fitness Home - TinyLlama QLoRA Training")
    print("=" * 76)
    print(f"Mode                    : {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")
    print(f"Base model              : {BASE_MODEL_NAME}")
    print(f"Train samples           : {len(train_records)}")
    print(f"Validation samples      : {len(validation_records)}")
    print(f"Reserved test samples   : {len(test_records)}")
    print(f"Run directory           : {run_dir}")
    print(f"CUDA device             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")

    tokenizer = load_tokenizer()

    train_dataset = FitnessHomeCausalDataset(
        records=train_records,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        split_name="train",
    )
    validation_dataset = FitnessHomeCausalDataset(
        records=validation_records,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        split_name="validation",
    )

    token_audit = dataset_audit(
        train_dataset,
        validation_dataset,
    )

    print("Tokenization audit:")
    print(json.dumps(token_audit, indent=2))

    model = load_qlora_model()

    training_args = build_training_arguments(
        output_dir=checkpoint_dir,
        smoke_test=args.smoke_test,
    )

    collator = CausalLMCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        collator=collator,
    )

    resume_checkpoint: Optional[str] = None

    if args.resume:
        checkpoint = newest_checkpoint(checkpoint_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint was found in {checkpoint_dir}."
            )
        resume_checkpoint = str(checkpoint)
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    start_time = time.time()

    train_result = trainer.train(
        resume_from_checkpoint=resume_checkpoint,
    )

    train_metrics = dict(train_result.metrics)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    validation_metrics = trainer.evaluate(
        eval_dataset=validation_dataset,
        metric_key_prefix="validation",
    )
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(adapter_dir)

    elapsed_seconds = time.time() - start_time

    save_run_metadata(
        path=metadata_file,
        smoke_test=args.smoke_test,
        run_dir=run_dir,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        token_audit=token_audit,
        elapsed_seconds=elapsed_seconds,
        adapter_dir=adapter_dir,
    )

    print()
    print("=" * 76)
    print("Training completed")
    print("=" * 76)
    print(f"Adapter directory       : {adapter_dir}")
    print(f"Training metadata       : {metadata_file}")
    print(f"Elapsed minutes         : {elapsed_seconds / 60:.2f}")
    print(f"Final validation loss   : {validation_metrics.get('validation_loss')}")
    print("Test split used         : NO")


if __name__ == "__main__":
    main()
