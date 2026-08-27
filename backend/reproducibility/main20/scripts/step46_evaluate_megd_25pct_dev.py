from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEED = 42
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 180
DEFAULT_BATCH_SIZE = 4
EXPECTED_VALIDATION_SAMPLES = 1948
EXPECTED_TEST_SAMPLES = 2069

SYSTEM_PROMPT = """
You are the Fitness Home recommendation explanation model.
The retrieval system has already selected the restaurant.
Use only the supplied evidence and constraint evaluation.
Do not recommend a different restaurant, invent facts, alter numbers,
or infer unsupported health benefits. Clearly state unmet constraints.
Return one concise evidence-grounded paragraph only.
""".strip()

SCRIPT_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_FILE.parent
DATASET_DIR = EXPERIMENT_ROOT / "04_main20k_split"
VALIDATION_FILE = DATASET_DIR / "validation.jsonl"
TEST_FILE = DATASET_DIR / "test.jsonl"
ADAPTER_DIR = (
    EXPERIMENT_ROOT
    / "25_main20k_qlora_megd_25pct"
    / "full_run_frozen"
    / "final_adapter"
)
OUTPUT_ROOT = EXPERIMENT_ROOT / "27_megd_25pct_dev_eval"

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

LEAKAGE_MARKERS = (
    "system prompt",
    "user request:",
    "selected restaurant evidence:",
    "constraint evaluation:",
    "overall match:",
    "you are the fitness home",
    "generate an evidence-grounded",
)

UNSUPPORTED_CLAIM_MARKERS = (
    "muscle repair",
    "muscle recovery",
    "support muscle growth",
    "supports muscle growth",
    "digestive health",
    "satiety",
    "traditional dining experience",
    "relatively healthy",
    "healthy option",
    "supports fat loss",
    "supporting fat loss",
    "boost metabolism",
    "improve metabolism",
    "supports recovery",
    "recovery benefits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen TinyLlama base and LoRA models."
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Validation smoke test only; forbidden for the test split.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalise_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalise_match(value: Any) -> str:
    return normalise_space(value).lower().replace("’", "'").replace("‘", "'")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}, line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_of(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def constraints_of(record: Dict[str, Any]) -> Dict[str, Any]:
    constraints = metadata_of(record).get("constraints")
    return constraints if isinstance(constraints, dict) else {}


def checks_of(record: Dict[str, Any]) -> Dict[str, bool]:
    checks = metadata_of(record).get("constraint_checks")
    return checks if isinstance(checks, dict) else {}


def validate_records(records: Sequence[Dict[str, Any]], split: str, limited: bool) -> None:
    expected = EXPECTED_VALIDATION_SAMPLES if split == "validation" else EXPECTED_TEST_SAMPLES
    if not limited and len(records) != expected:
        raise RuntimeError(f"Unexpected {split} sample count: expected {expected}, found {len(records)}.")

    sample_ids: List[str] = []
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise RuntimeError("A record has no sample_id.")
        sample_ids.append(sample_id)
        if metadata_of(record).get("filter_v2_3_accepted") is not True:
            raise RuntimeError(f"Non-accepted record found: {sample_id}.")
        for field in ("instruction", "input", "output"):
            if not str(record.get(field, "")).strip():
                raise RuntimeError(f"Sample {sample_id} has empty field: {field}.")

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"Duplicate sample IDs found in {split} split.")


def build_user_content(record: Dict[str, Any]) -> str:
    instruction = str(record["instruction"]).strip()
    evidence_input = str(record["input"]).strip()
    redundant_marker = "\n\nWrite one evidence-grounded recommendation explanation."
    if redundant_marker in evidence_input:
        evidence_input = evidence_input.split(redundant_marker, 1)[0].strip()
    return f"{instruction}\n\n{evidence_input}"


def build_prompt_text(tokenizer: Any, record: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(record)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError(f"Empty prompt for {record.get('sample_id', 'unknown')}.")
    return prompt


def load_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def quantization_config() -> BitsAndBytesConfig:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_base_model() -> Any:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config(),
        device_map={"": 0},
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


def load_lora_model() -> Any:
    base_model = load_base_model()
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        is_trainable=False,
    )
    model.config.use_cache = True
    model.eval()
    return model


def release_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def batches(records: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def generate_records(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    batch_size: int,
    model_label: str,
) -> Tuple[List[Dict[str, Any]], float]:
    outputs: List[Dict[str, Any]] = []
    start_time = time.time()
    completed = 0

    for batch in batches(records, batch_size):
        prompts = [build_prompt_text(tokenizer, record) for record in batch]
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
        encoded = {key: value.to(device) for key, value in encoded.items()}

        batch_start = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
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

        for record, prediction in zip(batch, decoded):
            completed += 1
            outputs.append(
                {
                    "sample_id": record["sample_id"],
                    "query_id": metadata_of(record).get("query_id"),
                    "model": model_label,
                    "prediction": normalise_space(prediction),
                    "reference": normalise_space(record["output"]),
                    "metadata": metadata_of(record),
                    "generation_seconds_estimate": round(batch_seconds / len(batch), 6),
                }
            )
            print(f"[{model_label}] {completed:03d}/{len(records):03d} {record['sample_id']}")

    return outputs, time.time() - start_time


def word_tokens(text: str) -> List[str]:
    return WORD_RE.findall(normalise_match(text))


def token_f1(prediction: str, reference: str) -> float:
    pred = word_tokens(prediction)
    ref = word_tokens(reference)
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    previous = [0] * (len(second) + 1)
    for token_first in first:
        current = [0]
        for index, token_second in enumerate(second, start=1):
            if token_first == token_second:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = word_tokens(prediction)
    ref = word_tokens(reference)
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def actual_values(record: Dict[str, Any]) -> Dict[str, int | None]:
    text = str(record.get("input", ""))
    patterns = {
        "calories": r"- Average calories:\s*(\d+(?:\.\d+)?)\s*kcal",
        "protein": r"- Average protein:\s*(\d+(?:\.\d+)?)\s*g",
        "fiber": r"- Average fibre:\s*(\d+(?:\.\d+)?)\s*g",
    }
    result: Dict[str, int | None] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        result[name] = safe_int(match.group(1)) if match else None
    return result


def allowed_numbers(record: Dict[str, Any]) -> set[str]:
    allowed = extract_numbers(str(record.get("input", "")))
    restaurant = str(metadata_of(record).get("restaurant_name", ""))
    allowed.update(extract_numbers(restaurant))

    constraints = constraints_of(record)
    observed = actual_values(record)
    pairs = [
        (observed.get("calories"), safe_int(constraints.get("max_calories"))),
        (observed.get("protein"), safe_int(constraints.get("min_protein"))),
        (observed.get("fiber"), safe_int(constraints.get("min_fiber"))),
    ]
    for actual, target in pairs:
        if actual is not None and target is not None:
            allowed.add(str(abs(actual - target)))
    allowed.update({"1", "2", "3", "4", "5"})
    return allowed


def restaurant_mentioned(prediction: str, restaurant_name: str) -> bool:
    prediction_norm = normalise_match(prediction)
    restaurant_norm = normalise_match(restaurant_name)
    if restaurant_norm and restaurant_norm in prediction_norm:
        return True
    base_name = re.sub(r"\s*\([^)]*\)\s*$", "", restaurant_norm).strip()
    return bool(len(base_name) >= 4 and base_name in prediction_norm)


def keyword_coverage(prediction: str, record: Dict[str, Any]) -> Tuple[int, int, int, int]:
    lower = normalise_match(prediction)
    constraints = constraints_of(record)
    checks = checks_of(record)
    required = ["cuisine", "calories", "protein"]
    if constraints.get("min_fiber") is not None:
        required.append("fiber")

    keywords = {
        "calories": ("calorie", "calories", "kcal"),
        "protein": ("protein",),
        "fiber": ("fiber", "fibre"),
    }

    covered = 0
    for name in required:
        if name == "cuisine":
            cuisine = normalise_match(constraints.get("cuisine", ""))
            covered += int(bool(cuisine and cuisine in lower))
        else:
            covered += int(any(keyword in lower for keyword in keywords[name]))

    failed = [name for name, passed in checks.items() if not bool(passed)]
    failed_covered = 0
    for name in failed:
        if name == "cuisine":
            cuisine = normalise_match(constraints.get("cuisine", ""))
            failed_covered += int(bool(cuisine and cuisine in lower))
        else:
            failed_covered += int(any(keyword in lower for keyword in keywords.get(name, (name,))))

    return covered, len(required), failed_covered, len(failed)


def analyse_prediction(prediction: str, reference: str, record: Dict[str, Any]) -> Dict[str, Any]:
    prediction = normalise_space(prediction)
    lower = normalise_match(prediction)
    words = word_tokens(prediction)
    unsupported_numbers = sorted(extract_numbers(prediction) - allowed_numbers(record))
    prompt_leakage = any(marker in lower for marker in LEAKAGE_MARKERS)
    unsupported_claim = any(marker in lower for marker in UNSUPPORTED_CLAIM_MARKERS)
    restaurant_ok = restaurant_mentioned(
        prediction,
        str(metadata_of(record).get("restaurant_name", "")),
    )
    one_paragraph = "\n\n" not in prediction
    length_ok = 20 <= len(words) <= 180
    format_success = bool(prediction) and one_paragraph and length_ok and not prompt_leakage
    numeric_faithful = not unsupported_numbers
    hallucination = bool(unsupported_numbers) or unsupported_claim or prompt_leakage or not restaurant_ok
    faithfulness_pass = restaurant_ok and numeric_faithful and not unsupported_claim and not prompt_leakage
    covered, total, failed_covered, failed_total = keyword_coverage(prediction, record)

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
        "constraint_coverage_rate": round(covered / total, 6) if total else 1.0,
        "failed_constraint_coverage_rate": round(failed_covered / failed_total, 6) if failed_total else 1.0,
        "reference_token_f1": round(token_f1(prediction, reference), 6),
        "rouge_l_f1": round(rouge_l_f1(prediction, reference), 6),
    }


def score_outputs(
    generated: Sequence[Dict[str, Any]],
    source_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for item in generated:
        source = source_by_id[str(item["sample_id"])]
        scored.append(
            {
                **item,
                "metrics": analyse_prediction(
                    str(item["prediction"]),
                    str(item["reference"]),
                    source,
                ),
            }
        )
    return scored


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rate(metrics: Sequence[Dict[str, Any]], key: str) -> float:
    return round(sum(bool(item[key]) for item in metrics) / len(metrics), 6) if metrics else 0.0


def summarise(scored: Sequence[Dict[str, Any]], generation_seconds: float) -> Dict[str, Any]:
    metrics = [item["metrics"] for item in scored]
    return {
        "sample_count": len(metrics),
        "format_success_rate": rate(metrics, "format_success"),
        "faithfulness_rate": rate(metrics, "faithfulness_pass"),
        "hallucination_rate": rate(metrics, "hallucination"),
        "restaurant_mention_accuracy": rate(metrics, "restaurant_mentioned"),
        "numeric_faithfulness_rate": rate(metrics, "numeric_faithful"),
        "prompt_leakage_rate": rate(metrics, "prompt_leakage"),
        "unsupported_health_or_goal_claim_rate": rate(metrics, "unsupported_health_or_goal_claim"),
        "mean_constraint_coverage_rate": round(mean([float(item["constraint_coverage_rate"]) for item in metrics]), 6),
        "mean_failed_constraint_coverage_rate": round(mean([float(item["failed_constraint_coverage_rate"]) for item in metrics]), 6),
        "mean_reference_token_f1": round(mean([float(item["reference_token_f1"]) for item in metrics]), 6),
        "mean_rouge_l_f1": round(mean([float(item["rouge_l_f1"]) for item in metrics]), 6),
        "mean_word_count": round(mean([float(item["word_count"]) for item in metrics]), 3),
        "generation_seconds": round(generation_seconds, 3),
        "samples_per_second": round(len(metrics) / generation_seconds, 6) if generation_seconds else 0.0,
    }


def write_paired_csv(
    path: Path,
    records: Sequence[Dict[str, Any]],
    base_scored: Sequence[Dict[str, Any]],
    lora_scored: Sequence[Dict[str, Any]],
) -> None:
    source_by_id = {str(item["sample_id"]): item for item in records}
    base_by_id = {str(item["sample_id"]): item for item in base_scored}
    lora_by_id = {str(item["sample_id"]): item for item in lora_scored}
    fieldnames = [
        "sample_id", "query_id", "match_type", "restaurant_name", "query",
        "reference", "base_prediction", "lora_prediction",
        "base_format_success", "lora_format_success",
        "base_faithfulness_pass", "lora_faithfulness_pass",
        "base_hallucination", "lora_hallucination",
        "base_constraint_coverage_rate", "lora_constraint_coverage_rate",
        "base_failed_constraint_coverage_rate", "lora_failed_constraint_coverage_rate",
        "base_reference_token_f1", "lora_reference_token_f1",
        "base_rouge_l_f1", "lora_rouge_l_f1",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(source_by_id):
            source = source_by_id[sample_id]
            metadata = metadata_of(source)
            base = base_by_id[sample_id]
            lora = lora_by_id[sample_id]
            bm = base["metrics"]
            lm = lora["metrics"]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "query_id": metadata.get("query_id"),
                    "match_type": metadata.get("match_type"),
                    "restaurant_name": metadata.get("restaurant_name"),
                    "query": metadata.get("query"),
                    "reference": source["output"],
                    "base_prediction": base["prediction"],
                    "lora_prediction": lora["prediction"],
                    "base_format_success": bm["format_success"],
                    "lora_format_success": lm["format_success"],
                    "base_faithfulness_pass": bm["faithfulness_pass"],
                    "lora_faithfulness_pass": lm["faithfulness_pass"],
                    "base_hallucination": bm["hallucination"],
                    "lora_hallucination": lm["hallucination"],
                    "base_constraint_coverage_rate": bm["constraint_coverage_rate"],
                    "lora_constraint_coverage_rate": lm["constraint_coverage_rate"],
                    "base_failed_constraint_coverage_rate": bm["failed_constraint_coverage_rate"],
                    "lora_failed_constraint_coverage_rate": lm["failed_constraint_coverage_rate"],
                    "base_reference_token_f1": bm["reference_token_f1"],
                    "lora_reference_token_f1": lm["reference_token_f1"],
                    "base_rouge_l_f1": bm["rouge_l_f1"],
                    "lora_rouge_l_f1": lm["rouge_l_f1"],
                }
            )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.split == "test" and args.limit is not None:
        raise ValueError("Do not use --limit with the final test split.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")

    required = [
        VALIDATION_FILE,
        TEST_FILE,
        ADAPTER_DIR / "adapter_config.json",
        ADAPTER_DIR / "adapter_model.safetensors",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact not found: {path}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    source_file = VALIDATION_FILE if args.split == "validation" else TEST_FILE
    all_records = read_jsonl(source_file)
    validate_records(all_records, args.split, limited=False)
    records = list(all_records[: args.limit]) if args.limit is not None else list(all_records)
    validate_records(records, args.split, limited=args.limit is not None)

    run_name = (
        f"{args.split}_smoke_{len(records)}"
        if args.limit is not None
        else f"{args.split}_final_{len(records)}"
    )
    run_dir = OUTPUT_ROOT / run_name
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("Fitness Home - Base vs LoRA Evaluation")
    print("=" * 76)
    print(f"Split                   : {args.split}")
    print(f"Samples                 : {len(records)}")
    print(f"Base model              : {BASE_MODEL_NAME}")
    print(f"Adapter                 : {ADAPTER_DIR}")
    print(f"Batch size              : {args.batch_size}")
    print(f"CUDA device             : {torch.cuda.get_device_name(0)}")
    print(f"Output directory        : {run_dir}")

    tokenizer = load_tokenizer()
    source_by_id = {str(record["sample_id"]): record for record in records}

    print("\n" + "=" * 76)
    print("Evaluating TinyLlama Base")
    print("=" * 76)
    base_model = load_base_model()
    base_generated, base_seconds = generate_records(
        base_model, tokenizer, records, args.batch_size, "tinyllama_base"
    )
    base_scored = score_outputs(base_generated, source_by_id)
    base_summary = summarise(base_scored, base_seconds)
    release_model(base_model)

    print("\n" + "=" * 76)
    print("Evaluating TinyLlama + LoRA")
    print("=" * 76)
    lora_model = load_lora_model()
    lora_generated, lora_seconds = generate_records(
        lora_model, tokenizer, records, args.batch_size, "tinyllama_lora"
    )
    lora_scored = score_outputs(lora_generated, source_by_id)
    lora_summary = summarise(lora_scored, lora_seconds)
    release_model(lora_model)

    base_file = run_dir / "base_predictions_scored.jsonl"
    lora_file = run_dir / "lora_predictions_scored.jsonl"
    paired_file = run_dir / "paired_predictions.csv"
    summary_file = run_dir / "evaluation_summary.json"
    checksum_file = run_dir / "SHA256SUMS.txt"

    write_jsonl(base_file, base_scored)
    write_jsonl(lora_file, lora_scored)
    write_paired_csv(paired_file, records, base_scored, lora_scored)

    compared: Dict[str, Any] = {}
    for key in sorted(set(base_summary) | set(lora_summary)):
        if key == "sample_count":
            continue
        base_value = float(base_summary.get(key, 0.0))
        lora_value = float(lora_summary.get(key, 0.0))
        compared[key] = {
            "base": base_value,
            "lora": lora_value,
            "absolute_change_lora_minus_base": round(lora_value - base_value, 6),
        }

    summary = {
        "experiment": "fitness_home_tinyllama_base_vs_lora_v1",
        "split": args.split,
        "is_final_test_evaluation": args.split == "test",
        "sample_count": len(records),
        "seed": SEED,
        "base_model": BASE_MODEL_NAME,
        "adapter_directory": str(ADAPTER_DIR),
        "generation_configuration": {
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": args.batch_size,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
        },
        "source_file": str(source_file),
        "source_sha256": sha256_file(source_file),
        "base": base_summary,
        "lora": lora_summary,
        "comparison": compared,
        "metric_notes": {
            "faithfulness_pass": "Selected restaurant mentioned; no unsupported number, prompt leakage, or flagged unsupported health/goal claim.",
            "hallucination": "Unsupported number, missing selected restaurant mention, prompt leakage, or flagged unsupported health/goal claim.",
            "reference_metrics": "Token F1 and ROUGE-L F1 against the frozen Teacher reference are supporting metrics, not standalone factuality metrics.",
        },
        "files": {
            "base_predictions": str(base_file),
            "lora_predictions": str(lora_file),
            "paired_predictions": str(paired_file),
        },
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with checksum_file.open("w", encoding="utf-8") as file:
        for path in (base_file, lora_file, paired_file, summary_file):
            file.write(f"{sha256_file(path)}  {path.name}\n")

    print("\n" + "=" * 76)
    print("Evaluation completed")
    print("=" * 76)
    print(f"Base faithfulness rate : {base_summary['faithfulness_rate']:.2%}")
    print(f"LoRA faithfulness rate : {lora_summary['faithfulness_rate']:.2%}")
    print(f"Base hallucination rate: {base_summary['hallucination_rate']:.2%}")
    print(f"LoRA hallucination rate: {lora_summary['hallucination_rate']:.2%}")
    print(f"Base format success    : {base_summary['format_success_rate']:.2%}")
    print(f"LoRA format success    : {lora_summary['format_success_rate']:.2%}")
    print(f"Base ROUGE-L F1        : {base_summary['mean_rouge_l_f1']:.4f}")
    print(f"LoRA ROUGE-L F1        : {lora_summary['mean_rouge_l_f1']:.4f}")
    print(f"Summary file           : {summary_file}")
    print(f"Final test split used  : {'YES' if args.split == 'test' else 'NO'}")


if __name__ == "__main__":
    main()
