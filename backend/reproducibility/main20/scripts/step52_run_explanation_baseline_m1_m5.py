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
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

HERE = Path(__file__).resolve().parent

PROTOCOL_DIR = HERE / "40_explanation_baseline_protocol"
PROTOCOL_FILE = PROTOCOL_DIR / "explanation_baseline_protocol.json"
METHOD_FILE = PROTOCOL_DIR / "method_manifest.json"
PROMPT_FILE = PROTOCOL_DIR / "prompt_templates.json"

DEV_FILE = (
    HERE
    / "19_eval_protocol"
    / "development_benchmark_2069.jsonl"
)
BLIND_SIGNATURE_FILE = (
    HERE
    / "19_eval_protocol"
    / "reserved_blind_signatures_250.jsonl"
)
LORA_ADAPTER_DIR = (
    HERE
    / "05_main20k_qlora_100pct"
    / "full_run_frozen"
    / "final_adapter"
)
FILTER_FILE = (
    HERE.parent
    / "01_filter_v2"
    / "step14_filter_v2_3_calibration.py"
)

OUTPUT_ROOT = HERE / "41_explanation_baseline_eval"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SEED = 42
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 180
DEFAULT_TINY_BATCH = 8
DEFAULT_TEACHER_BATCH = 4
NEAR_BOUNDARY_THRESHOLD = 0.10

METHOD_NAMES = {
    "M1": "Structured Rule/Template",
    "M2": "Base TinyLlama without RAG evidence",
    "M3": "Dense RAG + Base TinyLlama",
    "M4": "Dense RAG + LoRA TinyLlama",
    "M5": "Dense RAG + Llama-3.1-8B-Instruct",
}

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

LEAKAGE_MARKERS = (
    "system prompt",
    "user request:",
    "selected restaurant evidence:",
    "constraint evaluation:",
    "overall match:",
    "you are the fitness home",
    "generate an evidence-grounded",
    "teacher-v4 reminder",
)

UNSUPPORTED_CLAIM_MARKERS = (
    "muscle repair",
    "muscle recovery",
    "support muscle growth",
    "supports muscle growth",
    "supporting muscle gain",
    "digestive health",
    "satiety",
    "traditional dining experience",
    "relatively healthy",
    "healthy option",
    "supports fat loss",
    "supporting fat loss",
    "suitable for fat loss",
    "suitable for weight maintenance",
    "suitable for muscle gain",
    "suitable for post-workout recovery",
    "boost metabolism",
    "improve metabolism",
    "supports recovery",
    "recovery benefits",
    "beneficial for overall health",
)

AVERAGE_PATTERNS = {
    "calories": re.compile(
        r"- Average calories:\s*(\d+(?:\.\d+)?)\s*kcal",
        re.IGNORECASE,
    ),
    "protein": re.compile(
        r"- Average protein:\s*(\d+(?:\.\d+)?)\s*g",
        re.IGNORECASE,
    ),
    "fiber": re.compile(
        r"- Average (?:fibre|fiber):\s*(\d+(?:\.\d+)?)\s*g",
        re.IGNORECASE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen M1-M5 Fitness Home explanation baseline "
            "and produce paper-ready main, subgroup, and significance tables."
        )
    )
    parser.add_argument(
        "--methods",
        default="M1,M2,M3,M4,M5",
        help="Comma-separated subset of M1,M2,M3,M4,M5.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test limit. Full run uses all 2069 development samples.",
    )
    parser.add_argument(
        "--tinyllama-batch-size",
        type=int,
        default=DEFAULT_TINY_BATCH,
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=DEFAULT_TEACHER_BATCH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the selected run directory before starting.",
    )
    return parser.parse_args()


def normalise_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalise_text(value: Any) -> str:
    return (
        normalise_space(value)
        .lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("–", "-")
        .replace("—", "-")
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metadata_of(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata", {})
    return value if isinstance(value, dict) else {}


def constraints_of(record: dict[str, Any]) -> dict[str, Any]:
    value = metadata_of(record).get("constraints", {})
    return value if isinstance(value, dict) else {}


def checks_of(record: dict[str, Any]) -> dict[str, bool]:
    raw = metadata_of(record).get("constraint_checks", {})
    if not isinstance(raw, dict):
        return {}
    return {
        ("fiber" if key == "fibre" else str(key)): bool(value)
        for key, value in raw.items()
    }


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_records(records: Sequence[dict[str, Any]]) -> None:
    if len(records) != 2069:
        raise RuntimeError(
            f"Expected 2069 frozen development records, found {len(records)}."
        )

    sample_ids: list[str] = []
    signatures: set[str] = set()

    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise RuntimeError("A development record has no sample_id.")
        sample_ids.append(sample_id)

        metadata = metadata_of(record)
        if metadata.get("filter_v2_3_accepted") is not True:
            raise RuntimeError(
                f"Development record did not pass frozen Filter v2.3: {sample_id}"
            )

        signature = str(
            metadata.get("constraint_signature_id", "")
        ).strip()
        if not signature:
            raise RuntimeError(
                f"Missing constraint_signature_id: {sample_id}"
            )
        signatures.add(signature)

        for field in ("instruction", "input", "output"):
            if not str(record.get(field, "")).strip():
                raise RuntimeError(
                    f"Sample {sample_id} has empty field: {field}"
                )

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate development sample IDs detected.")

    if len(signatures) != 63:
        raise RuntimeError(
            f"Expected 63 development signatures, found {len(signatures)}."
        )


def split_sentences(text: str) -> list[str]:
    return [
        normalise_text(sentence)
        for sentence in SENTENCE_RE.split(str(text or ""))
        if normalise_space(sentence)
    ]


def number_string(value: Any) -> str | None:
    try:
        if value is None or value == "":
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def extract_number_strings(text: str) -> set[str]:
    values: set[str] = set()
    for token in NUMBER_RE.findall(text):
        try:
            numeric = float(token)
            values.add(str(int(numeric)) if numeric.is_integer() else f"{numeric:g}")
        except ValueError:
            continue
    return values


def parse_actual_values(record: dict[str, Any]) -> dict[str, float | None]:
    text = str(record.get("input", ""))
    values: dict[str, float | None] = {}

    for name, pattern in AVERAGE_PATTERNS.items():
        match = pattern.search(text)
        values[name] = float(match.group(1)) if match else None

    return values


def required_constraints(record: dict[str, Any]) -> list[str]:
    constraints = constraints_of(record)
    names = ["cuisine", "calories", "protein"]
    if constraints.get("min_fiber") is not None:
        names.append("fiber")
    return names


def failed_constraints(record: dict[str, Any]) -> list[str]:
    checks = checks_of(record)
    return [
        name
        for name in required_constraints(record)
        if checks.get(name) is False
    ]


def build_evidence_user_content(record: dict[str, Any]) -> str:
    instruction = str(record["instruction"]).strip()
    evidence_input = str(record["input"]).strip()

    markers = (
        "\n\nWrite one evidence-grounded recommendation explanation.",
        "\n\nTeacher-v4 reminder:",
    )
    for marker in markers:
        if marker in evidence_input:
            evidence_input = evidence_input.split(marker, 1)[0].strip()

    return f"{instruction}\n\n{evidence_input}"


def build_no_rag_user_content(
    record: dict[str, Any],
    prompt_template: str,
) -> str:
    metadata = metadata_of(record)
    return prompt_template.format(
        query=str(metadata.get("query", "")).strip(),
        restaurant_name=str(
            metadata.get("restaurant_name", "")
        ).strip(),
    )


def build_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_content: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Tokenizer produced an empty prompt.")
    return prompt


def build_template_prediction(record: dict[str, Any]) -> str:
    metadata = metadata_of(record)
    constraints = constraints_of(record)
    checks = checks_of(record)
    actual = parse_actual_values(record)

    restaurant = str(metadata.get("restaurant_name", "")).strip()
    match_type = str(metadata.get("match_type", "")).lower()

    satisfied: list[str] = []
    failed: list[str] = []

    cuisine = str(constraints.get("cuisine", "")).strip()
    if checks.get("cuisine") is True:
        satisfied.append(
            f"The cuisine requirement for {cuisine} is satisfied."
        )
    else:
        failed.append(
            f"the cuisine requirement for {cuisine} is not satisfied."
        )

    actual_calories = number_string(actual.get("calories"))
    max_calories = number_string(constraints.get("max_calories"))
    if actual_calories is None or max_calories is None:
        raise RuntimeError(
            f"Missing calorie values for {record.get('sample_id')}"
        )
    if checks.get("calories") is True:
        satisfied.append(
            f"Its average calorie value of {actual_calories} kcal is "
            f"within the maximum of {max_calories} kcal."
        )
    else:
        failed.append(
            f"its average calorie value of {actual_calories} kcal exceeds "
            f"the maximum of {max_calories} kcal."
        )

    actual_protein = number_string(actual.get("protein"))
    min_protein = number_string(constraints.get("min_protein"))
    if actual_protein is None or min_protein is None:
        raise RuntimeError(
            f"Missing protein values for {record.get('sample_id')}"
        )
    if checks.get("protein") is True:
        satisfied.append(
            f"Its average protein value of {actual_protein} g meets "
            f"the minimum of {min_protein} g."
        )
    else:
        failed.append(
            f"its average protein value of {actual_protein} g is below "
            f"the minimum of {min_protein} g."
        )

    if constraints.get("min_fiber") is not None:
        actual_fiber = number_string(actual.get("fiber"))
        min_fiber = number_string(constraints.get("min_fiber"))
        if actual_fiber is None or min_fiber is None:
            raise RuntimeError(
                f"Missing fibre values for {record.get('sample_id')}"
            )
        if checks.get("fiber") is True:
            satisfied.append(
                f"Its average fibre value of {actual_fiber} g meets "
                f"the minimum of {min_fiber} g."
            )
        else:
            failed.append(
                f"its average fibre value of {actual_fiber} g is below "
                f"the minimum of {min_fiber} g."
            )

    if match_type == "full":
        opening = f"{restaurant} satisfies all stated requirements."
    elif match_type == "weak":
        opening = f"{restaurant} meets some of the stated requirements."
    else:
        opening = f"{restaurant} is a partial match."

    parts = [opening]
    parts.extend(satisfied)

    if failed:
        failure_text = " ".join(failed)
        failure_text = (
            failure_text[0].upper() + failure_text[1:]
            if failure_text
            else failure_text
        )
        parts.append(f"However, {failure_text}")

    return normalise_space(" ".join(parts))


def quantization_config() -> BitsAndBytesConfig:
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def configure_generation(model: Any) -> None:
    model.config.use_cache = True
    model.eval()

    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.repetition_penalty = 1.0


def load_tiny_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_ADAPTER_DIR,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_teacher_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_base_model(model_name: str) -> Any:
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map={"": 0},
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    configure_generation(model)
    return model


def load_lora_model() -> Any:
    base = load_base_model(BASE_MODEL)
    model = PeftModel.from_pretrained(
        base,
        LORA_ADAPTER_DIR,
        is_trainable=False,
    )
    configure_generation(model)
    return model


def release_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def batches(
    records: Sequence[dict[str, Any]],
    batch_size: int,
) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def load_existing_predictions(
    path: Path,
    allowed_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in allowed_ids:
            raise RuntimeError(
                f"Prediction file contains unknown sample_id: {sample_id}"
            )
        if sample_id in by_id:
            raise RuntimeError(
                f"Duplicate prediction for sample_id: {sample_id}"
            )
        by_id[sample_id] = row

    return by_id


def generate_method_predictions(
    method_id: str,
    model: Any,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    system_prompt: str,
    user_content_builder: Callable[[dict[str, Any]], str],
    batch_size: int,
    output_path: Path,
) -> None:
    allowed_ids = {str(record["sample_id"]) for record in records}
    existing = load_existing_predictions(
        output_path,
        allowed_ids,
    )
    pending = [
        record
        for record in records
        if str(record["sample_id"]) not in existing
    ]

    if not pending:
        print(
            f"[{method_id}] Predictions already complete; skipping."
        )
        return

    print(
        f"[{method_id}] Existing={len(existing)} "
        f"Pending={len(pending)} Batch={batch_size}"
    )

    completed = len(existing)

    for batch in batches(pending, batch_size):
        user_contents = [
            user_content_builder(record)
            for record in batch
        ]
        prompts = [
            build_chat_prompt(
                tokenizer,
                system_prompt,
                user_content,
            )
            for user_content in user_contents
        ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            add_special_tokens=False,
        )

        input_width = int(encoded["input_ids"].shape[1])
        device = next(model.parameters()).device
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        batch_start = time.time()

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        batch_seconds = time.time() - batch_start
        new_tokens = generated[:, input_width:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for record, prediction, user_content in zip(
            batch,
            decoded,
            user_contents,
        ):
            completed += 1
            append_jsonl(
                output_path,
                {
                    "sample_id": record["sample_id"],
                    "method_id": method_id,
                    "method_name": METHOD_NAMES[method_id],
                    "prediction": normalise_space(prediction),
                    "reference": normalise_space(record["output"]),
                    "user_content_sha256": sha256_text(user_content),
                    "generation_seconds_estimate": round(
                        batch_seconds / len(batch),
                        6,
                    ),
                },
            )
            print(
                f"[{method_id}] {completed:04d}/{len(records):04d} "
                f"{record['sample_id']}"
            )


def ensure_template_predictions(
    records: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = [
        {
            "sample_id": record["sample_id"],
            "method_id": "M1",
            "method_name": METHOD_NAMES["M1"],
            "prediction": build_template_prediction(record),
            "reference": normalise_space(record["output"]),
            "user_content_sha256": None,
            "generation_seconds_estimate": 0.0,
        }
        for record in records
    ]
    write_jsonl(output_path, rows)


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(normalise_text(text))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = word_tokens(prediction)
    reference_tokens = word_tokens(reference)

    if not prediction_tokens or not reference_tokens:
        return 0.0

    overlap = sum(
        (
            Counter(prediction_tokens)
            & Counter(reference_tokens)
        ).values()
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return (
        2 * precision * recall / (precision + recall)
    )


def lcs_length(
    first: Sequence[str],
    second: Sequence[str],
) -> int:
    previous = [0] * (len(second) + 1)

    for token_first in first:
        current = [0]

        for index, token_second in enumerate(
            second,
            start=1,
        ):
            if token_first == token_second:
                current.append(
                    previous[index - 1] + 1
                )
            else:
                current.append(
                    max(
                        previous[index],
                        current[index - 1],
                    )
                )

        previous = current

    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = word_tokens(prediction)
    reference_tokens = word_tokens(reference)

    if not prediction_tokens or not reference_tokens:
        return 0.0

    lcs = lcs_length(
        prediction_tokens,
        reference_tokens,
    )
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def restaurant_mentioned(
    prediction: str,
    restaurant_name: str,
) -> bool:
    prediction_norm = normalise_text(prediction)
    restaurant_norm = normalise_text(restaurant_name)

    if restaurant_norm and restaurant_norm in prediction_norm:
        return True

    base_name = re.sub(
        r"\s*\([^)]*\)\s*$",
        "",
        restaurant_norm,
    ).strip()

    return bool(
        len(base_name) >= 4
        and base_name in prediction_norm
    )


def allowed_numbers(record: dict[str, Any]) -> set[str]:
    allowed = extract_number_strings(
        str(record.get("input", ""))
    )
    allowed.update(
        extract_number_strings(
            str(
                metadata_of(record).get(
                    "restaurant_name",
                    "",
                )
            )
        )
    )

    actual = parse_actual_values(record)
    constraints = constraints_of(record)

    pairs = (
        (
            actual.get("calories"),
            constraints.get("max_calories"),
        ),
        (
            actual.get("protein"),
            constraints.get("min_protein"),
        ),
        (
            actual.get("fiber"),
            constraints.get("min_fiber"),
        ),
    )

    for observed, target in pairs:
        if observed is not None and target is not None:
            difference = abs(
                float(observed) - float(target)
            )
            formatted = number_string(difference)
            if formatted is not None:
                allowed.add(formatted)

    allowed.update({"1", "2", "3", "4", "5"})
    return allowed


def relevant_sentences(
    prediction: str,
    constraint_name: str,
    target_cuisine: str,
) -> list[str]:
    sentences = split_sentences(prediction)

    if constraint_name == "cuisine":
        target = normalise_text(target_cuisine)
        return [
            sentence
            for sentence in sentences
            if "cuisine" in sentence
            or (target and target in sentence)
        ]

    if constraint_name == "calories":
        return [
            sentence
            for sentence in sentences
            if "calorie" in sentence
            or "kcal" in sentence
        ]

    if constraint_name == "protein":
        return [
            sentence
            for sentence in sentences
            if "protein" in sentence
        ]

    return [
        sentence
        for sentence in sentences
        if "fiber" in sentence
        or "fibre" in sentence
    ]


def classify_sentence_state(
    sentence: str,
    constraint_name: str,
    target_cuisine: str,
) -> bool | None:
    sentence = normalise_text(sentence)

    generic_negative = (
        r"\bnot satisfied\b",
        r"\bnot met\b",
        r"\bdoes not meet\b",
        r"\bdoesn't meet\b",
        r"\bdo not meet\b",
        r"\bfails? to meet\b",
        r"\bfailed\b",
        r"\bconstraint is not satisfied\b",
        r"\brequirement is not satisfied\b",
    )

    if constraint_name == "cuisine":
        target = re.escape(
            normalise_text(target_cuisine)
        )
        negative_patterns = (
            *generic_negative,
            r"\bcuisine mismatch\b",
            r"\binstead of\b",
            r"\bdoes not include\b",
            r"\bdo not include\b",
            r"\bdoesn't include\b",
            rf"\bnot\s+(?:a|an)\s+{target}\b",
        )
        positive_patterns = (
            r"\bcuisine requirement\b.{0,40}\b(?:satisfied|met)\b",
            rf"\b(?:a|an)\s+{target}\b",
            rf"\b{target}\s+(?:restaurant|option|meal|cuisine)\b",
            rf"\bmatches?\b.{0,40}\b{target}\b",
        )
    elif constraint_name == "calories":
        negative_patterns = (
            *generic_negative,
            r"\bexceeds?\b",
            r"\babove\b.{0,30}\b(?:maximum|limit|cap)\b",
            r"\bover\b.{0,30}\b(?:maximum|limit|cap)\b",
            r"\bhigher than\b",
            r"\bbeyond\b.{0,30}\b(?:maximum|limit|cap)\b",
        )
        positive_patterns = (
            r"\bwithin\b.{0,30}\b(?:maximum|limit|cap)\b",
            r"\bunder\b.{0,30}\b(?:maximum|limit|cap)\b",
            r"\bbelow\b.{0,30}\b(?:maximum|limit|cap)\b",
            r"\bat or below\b",
            r"\bdoes not exceed\b",
            r"\bstays? below\b",
            r"\bcalorie requirement\b.{0,40}\b(?:satisfied|met)\b",
            r"\bcalorie limit\b.{0,30}\b(?:satisfied|met)\b",
        )
    else:
        negative_patterns = (
            *generic_negative,
            r"\bfalls? short\b",
            r"\bbelow\b.{0,35}\b(?:minimum|requirement|threshold)\b",
            r"\bunder\b.{0,35}\b(?:minimum|requirement|threshold)\b",
            r"\binsufficient\b",
        )
        positive_patterns = (
            r"\bmeets?\b.{0,35}\b(?:minimum|requirement|threshold)\b",
            r"\bsatisfies?\b.{0,35}\b(?:minimum|requirement|threshold)\b",
            r"\bat least\b",
            r"\babove\b.{0,35}\b(?:minimum|requirement|threshold)\b",
            r"\bexceeds?\b.{0,35}\b(?:minimum|requirement|threshold)\b",
            r"\bmore than\b.{0,35}\b(?:minimum|requirement|threshold)\b",
            r"\bsufficient\b",
            r"\breaches?\b.{0,35}\b(?:minimum|requirement|threshold)\b",
        )

    if any(
        re.search(pattern, sentence)
        for pattern in negative_patterns
    ):
        return False

    if any(
        re.search(pattern, sentence)
        for pattern in positive_patterns
    ):
        return True

    return None


def predict_constraint_state(
    prediction: str,
    record: dict[str, Any],
    constraint_name: str,
) -> tuple[bool | None, bool, bool]:
    cuisine = str(
        constraints_of(record).get("cuisine", "")
    )
    sentences = relevant_sentences(
        prediction,
        constraint_name,
        cuisine,
    )

    states = {
        state
        for sentence in sentences
        if (
            state := classify_sentence_state(
                sentence,
                constraint_name,
                cuisine,
            )
        )
        is not None
    }

    contradictory = len(states) > 1

    if contradictory:
        return None, True, bool(sentences)

    if len(states) == 1:
        return next(iter(states)), False, True

    lower = normalise_text(prediction)
    all_positive = bool(
        re.search(
            r"\b(?:satisfies|meets|fulfills)\b"
            r".{0,25}\ball\b.{0,25}\brequirements\b",
            lower,
        )
    ) and not bool(
        re.search(
            r"\b(?:does not|doesn't|not)\b"
            r".{0,25}\ball\b.{0,25}\brequirements\b",
            lower,
        )
    )

    if all_positive:
        return True, False, bool(sentences)

    return None, False, bool(sentences)


def derive_subgroups(record: dict[str, Any]) -> dict[str, bool]:
    metadata = metadata_of(record)
    checks = checks_of(record)
    constraints = constraints_of(record)
    actual = parse_actual_values(record)

    margins: list[float] = []

    numeric_pairs = (
        (
            actual.get("calories"),
            constraints.get("max_calories"),
        ),
        (
            actual.get("protein"),
            constraints.get("min_protein"),
        ),
        (
            actual.get("fiber"),
            constraints.get("min_fiber"),
        ),
    )

    for observed, threshold in numeric_pairs:
        if observed is None or threshold is None:
            continue
        denominator = max(
            abs(float(threshold)),
            1.0,
        )
        margins.append(
            abs(float(observed) - float(threshold))
            / denominator
        )

    failed_count = sum(
        checks.get(name) is False
        for name in required_constraints(record)
    )

    return {
        "all": True,
        "full": str(
            metadata.get("match_type", "")
        ).lower()
        == "full",
        "weak": str(
            metadata.get("match_type", "")
        ).lower()
        == "weak",
        "partial": str(
            metadata.get("match_type", "")
        ).lower()
        == "partial",
        "near_boundary": bool(
            margins
            and min(margins)
            <= NEAR_BOUNDARY_THRESHOLD
        ),
        "cuisine_mismatch": (
            checks.get("cuisine") is False
        ),
        "multiple_failures": failed_count >= 2,
    }


def analyse_prediction(
    prediction: str,
    reference: str,
    record: dict[str, Any],
    strict_filter: Any | None,
) -> dict[str, Any]:
    prediction = normalise_space(prediction)
    lower = normalise_text(prediction)
    words = word_tokens(prediction)

    prediction_numbers = extract_number_strings(
        prediction
    )
    unsupported_numbers = sorted(
        prediction_numbers
        - allowed_numbers(record)
    )

    prompt_leakage = any(
        marker in lower
        for marker in LEAKAGE_MARKERS
    )
    unsupported_claim = any(
        marker in lower
        for marker in UNSUPPORTED_CLAIM_MARKERS
    )
    restaurant_ok = restaurant_mentioned(
        prediction,
        str(
            metadata_of(record).get(
                "restaurant_name",
                "",
            )
        ),
    )

    one_paragraph = "\n\n" not in prediction
    length_ok = 20 <= len(words) <= 180
    format_success = (
        bool(prediction)
        and one_paragraph
        and length_ok
        and not prompt_leakage
    )

    numeric_faithful = not unsupported_numbers
    hallucination = bool(
        unsupported_numbers
        or unsupported_claim
        or prompt_leakage
        or not restaurant_ok
    )
    faithfulness_pass = bool(
        restaurant_ok
        and numeric_faithful
        and not unsupported_claim
        and not prompt_leakage
    )

    constraints = constraints_of(record)
    checks = checks_of(record)
    actual = parse_actual_values(record)

    state_details: dict[str, Any] = {}
    correct_states = 0
    mentioned_states = 0
    failed_correct = 0
    failed_total = 0
    numeric_correct = 0
    numeric_total = 0

    for name in required_constraints(record):
        predicted_state, contradictory, mentioned = (
            predict_constraint_state(
                prediction,
                record,
                name,
            )
        )
        truth = bool(checks[name])
        correct = (
            predicted_state is not None
            and predicted_state == truth
            and not contradictory
        )

        if mentioned:
            mentioned_states += 1
        if correct:
            correct_states += 1

        if truth is False:
            failed_total += 1
            if correct:
                failed_correct += 1

        actual_mentioned = None
        relation_correct = None

        if name in ("calories", "protein", "fiber"):
            numeric_total += 1
            actual_text = number_string(actual.get(name))
            actual_mentioned = bool(
                actual_text
                and actual_text in prediction_numbers
            )
            relation_correct = bool(
                actual_mentioned
                and correct
                and not contradictory
            )
            if relation_correct:
                numeric_correct += 1

        state_details[name] = {
            "truth": truth,
            "prediction": predicted_state,
            "mentioned": mentioned,
            "contradictory": contradictory,
            "correct": correct,
            "actual_value_mentioned": actual_mentioned,
            "numeric_relation_correct": relation_correct,
        }

    total_constraints = len(
        required_constraints(record)
    )

    all_constraint_exact = bool(
        correct_states == total_constraints
        and numeric_correct == numeric_total
        and restaurant_ok
        and numeric_faithful
        and not unsupported_claim
        and not prompt_leakage
    )

    strict_reasons: list[str] = []
    if strict_filter is not None:
        strict_record = json.loads(
            json.dumps(record, ensure_ascii=False)
        )
        strict_record["output"] = prediction
        strict_reasons = list(
            strict_filter.filter_reasons(
                strict_record
            )
        )

    return {
        "word_count": len(words),
        "format_success": format_success,
        "restaurant_mentioned": restaurant_ok,
        "unsupported_numbers": unsupported_numbers,
        "numeric_faithful": numeric_faithful,
        "prompt_leakage": prompt_leakage,
        "unsupported_health_or_goal_claim": unsupported_claim,
        "hallucination": hallucination,
        "faithfulness_pass": faithfulness_pass,
        "constraint_states": state_details,
        "constraint_state_correct_count": correct_states,
        "constraint_state_total_count": total_constraints,
        "constraint_state_coverage_rate": (
            mentioned_states / total_constraints
            if total_constraints
            else 1.0
        ),
        "failed_constraint_correct_count": failed_correct,
        "failed_constraint_total_count": failed_total,
        "failed_constraint_recall": (
            failed_correct / failed_total
            if failed_total
            else 1.0
        ),
        "numeric_relation_correct_count": numeric_correct,
        "numeric_relation_total_count": numeric_total,
        "numeric_relation_accuracy": (
            numeric_correct / numeric_total
            if numeric_total
            else 1.0
        ),
        "all_constraint_exact": all_constraint_exact,
        "reference_token_f1": token_f1(
            prediction,
            reference,
        ),
        "rouge_l_f1": rouge_l_f1(
            prediction,
            reference,
        ),
        "strict_filter_pass": not strict_reasons,
        "strict_filter_reasons": strict_reasons,
        "subgroups": derive_subgroups(record),
    }


def macro_f1_from_scored(
    scored: Sequence[dict[str, Any]],
) -> float:
    class_f1: list[float] = []

    for target_class in (True, False):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for item in scored:
            for detail in item["metrics"][
                "constraint_states"
            ].values():
                truth = detail["truth"]
                prediction = detail["prediction"]

                if prediction == target_class:
                    if truth == target_class:
                        true_positive += 1
                    else:
                        false_positive += 1
                elif truth == target_class:
                    false_negative += 1

        precision = (
            true_positive
            / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive
            / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_f1.append(f1)

    return sum(class_f1) / len(class_f1)


def aggregate_summary(
    scored: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not scored:
        return {"sample_count": 0}

    metrics = [item["metrics"] for item in scored]
    generation_seconds = sum(
        float(
            item.get(
                "generation_seconds_estimate",
                0.0,
            )
        )
        for item in scored
    )

    failed_correct = sum(
        int(metric["failed_constraint_correct_count"])
        for metric in metrics
    )
    failed_total = sum(
        int(metric["failed_constraint_total_count"])
        for metric in metrics
    )
    numeric_correct = sum(
        int(metric["numeric_relation_correct_count"])
        for metric in metrics
    )
    numeric_total = sum(
        int(metric["numeric_relation_total_count"])
        for metric in metrics
    )
    state_correct = sum(
        int(metric["constraint_state_correct_count"])
        for metric in metrics
    )
    state_total = sum(
        int(metric["constraint_state_total_count"])
        for metric in metrics
    )

    def boolean_rate(key: str) -> float:
        return sum(
            bool(metric[key])
            for metric in metrics
        ) / len(metrics)

    def mean_value(key: str) -> float:
        return sum(
            float(metric[key])
            for metric in metrics
        ) / len(metrics)

    return {
        "sample_count": len(metrics),
        "all_constraint_exact_accuracy": boolean_rate(
            "all_constraint_exact"
        ),
        "constraint_state_accuracy": (
            state_correct / state_total
            if state_total
            else 0.0
        ),
        "constraint_state_macro_f1": (
            macro_f1_from_scored(scored)
        ),
        "numeric_relation_accuracy": (
            numeric_correct / numeric_total
            if numeric_total
            else 0.0
        ),
        "failed_constraint_recall": (
            failed_correct / failed_total
            if failed_total
            else 1.0
        ),
        "restaurant_mention_accuracy": boolean_rate(
            "restaurant_mentioned"
        ),
        "faithfulness_rate": boolean_rate(
            "faithfulness_pass"
        ),
        "hallucination_rate": boolean_rate(
            "hallucination"
        ),
        "format_success_rate": boolean_rate(
            "format_success"
        ),
        "numeric_faithfulness_rate": boolean_rate(
            "numeric_faithful"
        ),
        "strict_filter_pass_rate": boolean_rate(
            "strict_filter_pass"
        ),
        "mean_constraint_state_coverage_rate": mean_value(
            "constraint_state_coverage_rate"
        ),
        "mean_reference_token_f1": mean_value(
            "reference_token_f1"
        ),
        "mean_rouge_l_f1": mean_value(
            "rouge_l_f1"
        ),
        "mean_word_count": mean_value(
            "word_count"
        ),
        "generation_seconds_estimate": generation_seconds,
        "samples_per_second_estimate": (
            len(metrics) / generation_seconds
            if generation_seconds
            else None
        ),
    }


def score_method(
    method_id: str,
    prediction_path: Path,
    records: Sequence[dict[str, Any]],
    strict_filter: Any | None,
    scored_path: Path,
) -> list[dict[str, Any]]:
    predictions = read_jsonl(prediction_path)
    prediction_by_id = {
        str(row["sample_id"]): row
        for row in predictions
    }
    source_by_id = {
        str(record["sample_id"]): record
        for record in records
    }

    if set(prediction_by_id) != set(source_by_id):
        raise RuntimeError(
            f"{method_id} prediction/source ID mismatch."
        )

    scored: list[dict[str, Any]] = []

    for sample_id in sorted(source_by_id):
        prediction_row = prediction_by_id[sample_id]
        source = source_by_id[sample_id]

        scored.append(
            {
                **prediction_row,
                "metadata": metadata_of(source),
                "metrics": analyse_prediction(
                    str(prediction_row["prediction"]),
                    str(source["output"]),
                    source,
                    strict_filter,
                ),
            }
        )

    write_jsonl(scored_path, scored)
    return scored


def subgroup_summaries(
    scored: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    subgroup_names = (
        "all",
        "full",
        "weak",
        "partial",
        "near_boundary",
        "cuisine_mismatch",
        "multiple_failures",
    )

    summaries: dict[str, dict[str, Any]] = {}

    for subgroup in subgroup_names:
        group = [
            item
            for item in scored
            if item["metrics"]["subgroups"].get(
                subgroup,
                False,
            )
        ]
        summaries[subgroup] = aggregate_summary(
            group
        )

    return summaries


def exact_mcnemar_p_value(
    first: Sequence[bool],
    second: Sequence[bool],
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("Paired lists have different lengths.")

    first_only = sum(
        a and not b
        for a, b in zip(first, second)
    )
    second_only = sum(
        b and not a
        for a, b in zip(first, second)
    )
    discordant = first_only + second_only

    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k)
            for k in range(
                0,
                min(first_only, second_only) + 1,
            )
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)

    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant_pairs": discordant,
        "two_sided_exact_mcnemar_p": p_value,
    }


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    seed: int,
    repetitions: int = 2000,
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("Paired lists have different lengths.")
    if not first:
        return {
            "mean_difference_second_minus_first": 0.0,
            "bootstrap_95ci": [0.0, 0.0],
            "repetitions": repetitions,
        }

    rng = random.Random(seed)
    differences = [
        float(b) - float(a)
        for a, b in zip(first, second)
    ]
    observed = sum(differences) / len(differences)

    bootstrap_means: list[float] = []

    for _ in range(repetitions):
        sample = [
            differences[
                rng.randrange(len(differences))
            ]
            for _ in range(len(differences))
        ]
        bootstrap_means.append(
            sum(sample) / len(sample)
        )

    bootstrap_means.sort()

    lower_index = int(
        0.025 * (repetitions - 1)
    )
    upper_index = int(
        0.975 * (repetitions - 1)
    )

    return {
        "mean_difference_second_minus_first": observed,
        "bootstrap_95ci": [
            bootstrap_means[lower_index],
            bootstrap_means[upper_index],
        ],
        "repetitions": repetitions,
    }


def pairwise_significance(
    first_id: str,
    second_id: str,
    scored_by_method: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    first_by_id = {
        str(item["sample_id"]): item
        for item in scored_by_method[first_id]
    }
    second_by_id = {
        str(item["sample_id"]): item
        for item in scored_by_method[second_id]
    }

    sample_ids = sorted(first_by_id)

    if sample_ids != sorted(second_by_id):
        raise RuntimeError(
            f"Paired sample mismatch: {first_id} vs {second_id}"
        )

    exact_first = [
        bool(
            first_by_id[sample_id]["metrics"][
                "all_constraint_exact"
            ]
        )
        for sample_id in sample_ids
    ]
    exact_second = [
        bool(
            second_by_id[sample_id]["metrics"][
                "all_constraint_exact"
            ]
        )
        for sample_id in sample_ids
    ]
    faithful_first = [
        bool(
            first_by_id[sample_id]["metrics"][
                "faithfulness_pass"
            ]
        )
        for sample_id in sample_ids
    ]
    faithful_second = [
        bool(
            second_by_id[sample_id]["metrics"][
                "faithfulness_pass"
            ]
        )
        for sample_id in sample_ids
    ]
    rouge_first = [
        float(
            first_by_id[sample_id]["metrics"][
                "rouge_l_f1"
            ]
        )
        for sample_id in sample_ids
    ]
    rouge_second = [
        float(
            second_by_id[sample_id]["metrics"][
                "rouge_l_f1"
            ]
        )
        for sample_id in sample_ids
    ]

    return {
        "comparison": f"{first_id}_vs_{second_id}",
        "direction": (
            f"All differences are {second_id} minus {first_id}."
        ),
        "all_constraint_exact": exact_mcnemar_p_value(
            exact_first,
            exact_second,
        ),
        "faithfulness": exact_mcnemar_p_value(
            faithful_first,
            faithful_second,
        ),
        "rouge_l": paired_bootstrap_difference(
            rouge_first,
            rouge_second,
            seed=SEED,
        ),
    }


def percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{100.0 * float(value):.2f}"


def decimal(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_main_table(
    path_csv: Path,
    path_md: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "Method",
        "All-Constraint Exact Accuracy (%)",
        "Constraint-State Macro-F1 (%)",
        "Numeric Relation Accuracy (%)",
        "Failed-Constraint Recall (%)",
        "Restaurant Mention Accuracy (%)",
        "Faithfulness (%)",
        "Hallucination (%)",
        "Format Success (%)",
        "ROUGE-L F1",
        "Token F1",
    ]

    rows: list[dict[str, str]] = []

    for method_id in ("M1", "M2", "M3", "M4", "M5"):
        if method_id not in summaries:
            continue
        summary = summaries[method_id]

        rows.append(
            {
                "Method": (
                    f"{method_id} {METHOD_NAMES[method_id]}"
                ),
                "All-Constraint Exact Accuracy (%)": percent(
                    summary[
                        "all_constraint_exact_accuracy"
                    ]
                ),
                "Constraint-State Macro-F1 (%)": percent(
                    summary[
                        "constraint_state_macro_f1"
                    ]
                ),
                "Numeric Relation Accuracy (%)": percent(
                    summary[
                        "numeric_relation_accuracy"
                    ]
                ),
                "Failed-Constraint Recall (%)": percent(
                    summary[
                        "failed_constraint_recall"
                    ]
                ),
                "Restaurant Mention Accuracy (%)": percent(
                    summary[
                        "restaurant_mention_accuracy"
                    ]
                ),
                "Faithfulness (%)": percent(
                    summary["faithfulness_rate"]
                ),
                "Hallucination (%)": percent(
                    summary["hallucination_rate"]
                ),
                "Format Success (%)": percent(
                    summary["format_success_rate"]
                ),
                "ROUGE-L F1": decimal(
                    summary["mean_rouge_l_f1"]
                ),
                "Token F1": decimal(
                    summary["mean_reference_token_f1"]
                ),
            }
        )

    with path_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(fieldnames) + " |"
    separator = (
        "| "
        + " | ".join(["---"] * len(fieldnames))
        + " |"
    )
    body = [
        "| "
        + " | ".join(row[name] for name in fieldnames)
        + " |"
        for row in rows
    ]

    path_md.write_text(
        "\n".join([header, separator, *body]) + "\n",
        encoding="utf-8",
    )


def write_subgroup_table(
    path: Path,
    subgroup_by_method: dict[
        str,
        dict[str, dict[str, Any]],
    ],
) -> None:
    fieldnames = [
        "Method",
        "Subgroup",
        "Samples",
        "All-Constraint Exact Accuracy (%)",
        "Constraint-State Macro-F1 (%)",
        "Numeric Relation Accuracy (%)",
        "Failed-Constraint Recall (%)",
        "Faithfulness (%)",
        "Hallucination (%)",
        "ROUGE-L F1",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for method_id in (
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
        ):
            if method_id not in subgroup_by_method:
                continue

            for subgroup, summary in (
                subgroup_by_method[method_id].items()
            ):
                writer.writerow(
                    {
                        "Method": method_id,
                        "Subgroup": subgroup,
                        "Samples": summary.get(
                            "sample_count",
                            0,
                        ),
                        "All-Constraint Exact Accuracy (%)": percent(
                            summary.get(
                                "all_constraint_exact_accuracy"
                            )
                        ),
                        "Constraint-State Macro-F1 (%)": percent(
                            summary.get(
                                "constraint_state_macro_f1"
                            )
                        ),
                        "Numeric Relation Accuracy (%)": percent(
                            summary.get(
                                "numeric_relation_accuracy"
                            )
                        ),
                        "Failed-Constraint Recall (%)": percent(
                            summary.get(
                                "failed_constraint_recall"
                            )
                        ),
                        "Faithfulness (%)": percent(
                            summary.get(
                                "faithfulness_rate"
                            )
                        ),
                        "Hallucination (%)": percent(
                            summary.get(
                                "hallucination_rate"
                            )
                        ),
                        "ROUGE-L F1": decimal(
                            summary.get(
                                "mean_rouge_l_f1"
                            )
                        ),
                    }
                )


def build_run_manifest(
    records: Sequence[dict[str, Any]],
    methods: Sequence[str],
    tiny_batch: int,
    teacher_batch: int,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "experiment": "fitness_home_explanation_baseline_m1_m5_v1",
        "status": (
            "development_benchmark_only_final_blind_not_used"
        ),
        "sample_count": len(records),
        "requested_methods": list(methods),
        "seed": SEED,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "tinyllama_batch_size": tiny_batch,
        "teacher_batch_size": teacher_batch,
        "limit": limit,
        "development_file": str(DEV_FILE),
        "development_sha256": sha256_file(DEV_FILE),
        "blind_signature_file": str(
            BLIND_SIGNATURE_FILE
        ),
        "blind_signature_sha256": sha256_file(
            BLIND_SIGNATURE_FILE
        ),
        "blind_test_used": False,
        "protocol_sha256": sha256_file(
            PROTOCOL_FILE
        ),
        "method_manifest_sha256": sha256_file(
            METHOD_FILE
        ),
        "prompt_templates_sha256": sha256_file(
            PROMPT_FILE
        ),
        "lora_adapter_model_sha256": sha256_file(
            LORA_ADAPTER_DIR
            / "adapter_model.safetensors"
        ),
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "metric_definitions": {
            "all_constraint_exact_accuracy": (
                "A sample is correct only when every required constraint "
                "state is correctly expressed, each numerical actual value "
                "is mentioned with the correct satisfaction relation, the "
                "selected restaurant is mentioned, and no unsupported number, "
                "unsupported health/goal claim, or prompt leakage is detected."
            ),
            "constraint_state_macro_f1": (
                "Macro-F1 across the satisfied and failed state classes "
                "over all required constraints; omitted or contradictory "
                "states count as errors."
            ),
            "numeric_relation_accuracy": (
                "Micro accuracy across required nutritional constraints; "
                "the actual database value must be mentioned and the "
                "satisfied/failed relation must match constraint_checks."
            ),
            "failed_constraint_recall": (
                "Micro recall of failed constraints that are explicitly "
                "and correctly described as failed."
            ),
            "near_boundary": (
                "At least one required numeric constraint has a relative "
                f"margin <= {NEAR_BOUNDARY_THRESHOLD:.2f}."
            ),
        },
        "important_note": (
            "M5 is a teacher/reference upper bound. Development references "
            "were generated by the same teacher family, so M5 ROUGE and "
            "Token-F1 are not treated as an independent fair comparison."
        ),
    }


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for M2-M5 evaluation."
        )
    if args.tinyllama_batch_size < 1:
        raise ValueError(
            "--tinyllama-batch-size must be >= 1."
        )
    if args.teacher_batch_size < 1:
        raise ValueError(
            "--teacher-batch-size must be >= 1."
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1.")

    requested_methods = [
        method.strip().upper()
        for method in args.methods.split(",")
        if method.strip()
    ]
    invalid = [
        method
        for method in requested_methods
        if method not in METHOD_NAMES
    ]
    if invalid:
        raise ValueError(
            f"Unknown methods: {invalid}"
        )
    requested_methods = list(
        dict.fromkeys(requested_methods)
    )

    required_files = (
        PROTOCOL_FILE,
        METHOD_FILE,
        PROMPT_FILE,
        DEV_FILE,
        BLIND_SIGNATURE_FILE,
        LORA_ADAPTER_DIR / "adapter_config.json",
        LORA_ADAPTER_DIR / "adapter_model.safetensors",
    )
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = read_json(PROTOCOL_FILE)
    prompts = read_json(PROMPT_FILE)

    if protocol.get("status") != (
        "frozen_before_baseline_execution"
    ):
        raise RuntimeError(
            "Baseline protocol is not in frozen status."
        )
    if protocol["benchmark"]["blind_test_used"] is not False:
        raise RuntimeError(
            "Frozen protocol unexpectedly marks blind test as used."
        )

    all_records = read_jsonl(DEV_FILE)
    validate_records(all_records)

    records = (
        all_records[: args.limit]
        if args.limit is not None
        else all_records
    )

    run_name = (
        f"smoke_{len(records)}"
        if args.limit is not None
        else "development_2069"
    )
    run_dir = OUTPUT_ROOT / run_name

    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        records,
        requested_methods,
        args.tinyllama_batch_size,
        args.teacher_batch_size,
        args.limit,
    )
    manifest_path = run_dir / "run_manifest.json"

    if manifest_path.exists():
        existing_manifest = read_json(
            manifest_path
        )
        if existing_manifest != manifest:
            raise RuntimeError(
                "Existing run manifest differs from current frozen "
                "configuration. Use --overwrite to start a new run."
            )
    else:
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    print("=" * 78)
    print("FITNESS HOME — FROZEN M1-M5 EXPLANATION BASELINE")
    print("=" * 78)
    print("Development samples :", len(records))
    print("Methods             :", requested_methods)
    print("TinyLlama batch     :", args.tinyllama_batch_size)
    print("Llama-8B batch      :", args.teacher_batch_size)
    print("CUDA device         :", torch.cuda.get_device_name(0))
    print("Blind test used     : NO")
    print("Output              :", run_dir)

    prediction_paths = {
        method_id: (
            run_dir
            / f"{method_id.lower()}_predictions.jsonl"
        )
        for method_id in requested_methods
    }

    if "M1" in requested_methods:
        existing = (
            read_jsonl(prediction_paths["M1"])
            if prediction_paths["M1"].exists()
            else []
        )
        if len(existing) != len(records):
            ensure_template_predictions(
                records,
                prediction_paths["M1"],
            )
        print("[M1] Template predictions complete.")

    system_prompt = prompts[
        "common_system_prompt"
    ]
    m2_prompt = prompts[
        "m2_no_rag_user_prompt"
    ]

    if any(
        method in requested_methods
        for method in ("M2", "M3")
    ):
        tiny_tokenizer = load_tiny_tokenizer()
        base_model = load_base_model(BASE_MODEL)

        if "M2" in requested_methods:
            generate_method_predictions(
                method_id="M2",
                model=base_model,
                tokenizer=tiny_tokenizer,
                records=records,
                system_prompt=system_prompt,
                user_content_builder=(
                    lambda record: build_no_rag_user_content(
                        record,
                        m2_prompt,
                    )
                ),
                batch_size=args.tinyllama_batch_size,
                output_path=prediction_paths["M2"],
            )

        if "M3" in requested_methods:
            generate_method_predictions(
                method_id="M3",
                model=base_model,
                tokenizer=tiny_tokenizer,
                records=records,
                system_prompt=system_prompt,
                user_content_builder=(
                    build_evidence_user_content
                ),
                batch_size=args.tinyllama_batch_size,
                output_path=prediction_paths["M3"],
            )

        release_model(base_model)
        del tiny_tokenizer
        gc.collect()

    if "M4" in requested_methods:
        tiny_tokenizer = load_tiny_tokenizer()
        lora_model = load_lora_model()

        generate_method_predictions(
            method_id="M4",
            model=lora_model,
            tokenizer=tiny_tokenizer,
            records=records,
            system_prompt=system_prompt,
            user_content_builder=(
                build_evidence_user_content
            ),
            batch_size=args.tinyllama_batch_size,
            output_path=prediction_paths["M4"],
        )

        release_model(lora_model)
        del tiny_tokenizer
        gc.collect()

    if "M5" in requested_methods:
        teacher_tokenizer = (
            load_teacher_tokenizer()
        )
        teacher_model = load_base_model(
            TEACHER_MODEL
        )

        generate_method_predictions(
            method_id="M5",
            model=teacher_model,
            tokenizer=teacher_tokenizer,
            records=records,
            system_prompt=system_prompt,
            user_content_builder=(
                build_evidence_user_content
            ),
            batch_size=args.teacher_batch_size,
            output_path=prediction_paths["M5"],
        )

        release_model(teacher_model)
        del teacher_tokenizer
        gc.collect()

    strict_filter = (
        load_module(
            FILTER_FILE,
            "fitness_home_filter_v23_baseline",
        )
        if FILTER_FILE.exists()
        else None
    )

    scored_by_method: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    summary_by_method: dict[
        str,
        dict[str, Any],
    ] = {}
    subgroup_by_method: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for method_id in requested_methods:
        prediction_path = prediction_paths[
            method_id
        ]
        predictions = read_jsonl(
            prediction_path
        )
        if len(predictions) != len(records):
            raise RuntimeError(
                f"{method_id} predictions incomplete: "
                f"{len(predictions)}/{len(records)}"
            )

        scored_path = (
            run_dir
            / f"{method_id.lower()}_predictions_scored.jsonl"
        )
        scored = score_method(
            method_id,
            prediction_path,
            records,
            strict_filter,
            scored_path,
        )
        scored_by_method[method_id] = scored
        summary_by_method[method_id] = (
            aggregate_summary(scored)
        )
        subgroup_by_method[method_id] = (
            subgroup_summaries(scored)
        )

    main_summary = {
        "experiment": (
            "fitness_home_explanation_baseline_m1_m5_v1"
        ),
        "development_only": True,
        "blind_test_used": False,
        "sample_count": len(records),
        "methods": summary_by_method,
        "subgroups": subgroup_by_method,
        "teacher_reference_bias_note": (
            "M5 uses the same teacher model family that generated "
            "the Development references; M5 ROUGE and Token-F1 "
            "are supporting upper-bound values rather than an "
            "independent fair baseline."
        ),
    }

    summary_path = (
        run_dir / "baseline_evaluation_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            main_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    main_csv = run_dir / "baseline_main_table.csv"
    main_md = run_dir / "baseline_main_table.md"
    write_main_table(
        main_csv,
        main_md,
        summary_by_method,
    )

    subgroup_csv = (
        run_dir / "baseline_subgroup_table.csv"
    )
    write_subgroup_table(
        subgroup_csv,
        subgroup_by_method,
    )

    planned_pairs = (
        ("M2", "M3"),
        ("M3", "M4"),
        ("M1", "M4"),
        ("M4", "M5"),
    )
    pairwise: list[dict[str, Any]] = []

    for first_id, second_id in planned_pairs:
        if (
            first_id in scored_by_method
            and second_id in scored_by_method
        ):
            pairwise.append(
                pairwise_significance(
                    first_id,
                    second_id,
                    scored_by_method,
                )
            )

    pairwise_path = (
        run_dir
        / "pairwise_significance.json"
    )
    pairwise_path.write_text(
        json.dumps(
            pairwise,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checksum_path = run_dir / "SHA256SUMS.txt"
    output_files = [
        manifest_path,
        summary_path,
        main_csv,
        main_md,
        subgroup_csv,
        pairwise_path,
        *[
            prediction_paths[method_id]
            for method_id in requested_methods
        ],
        *[
            run_dir
            / f"{method_id.lower()}_predictions_scored.jsonl"
            for method_id in requested_methods
        ],
    ]

    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for path in output_files:
            file.write(
                f"{sha256_file(path)}  {path.name}\n"
            )

    print()
    print("=" * 78)
    print("M1-M5 EXPLANATION BASELINE COMPLETE")
    print("=" * 78)

    for method_id in (
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ):
        if method_id not in summary_by_method:
            continue
        summary = summary_by_method[method_id]
        print(
            f"{method_id} "
            f"Exact={summary['all_constraint_exact_accuracy']:.2%} "
            f"StateF1={summary['constraint_state_macro_f1']:.2%} "
            f"Numeric={summary['numeric_relation_accuracy']:.2%} "
            f"FailedRecall={summary['failed_constraint_recall']:.2%} "
            f"Faith={summary['faithfulness_rate']:.2%} "
            f"Hallu={summary['hallucination_rate']:.2%} "
            f"ROUGE-L={summary['mean_rouge_l_f1']:.4f}"
        )

    print("Main table       :", main_csv)
    print("Markdown table   :", main_md)
    print("Subgroup table   :", subgroup_csv)
    print("Significance     :", pairwise_path)
    print("Summary          :", summary_path)
    print("Blind test used  : NO")


if __name__ == "__main__":
    main()
