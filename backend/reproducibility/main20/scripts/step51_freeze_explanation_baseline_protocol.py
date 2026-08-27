#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

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

OUT_DIR = HERE / "40_explanation_baseline_protocol"
PROTOCOL_FILE = OUT_DIR / "explanation_baseline_protocol.json"
METHOD_FILE = OUT_DIR / "method_manifest.json"
PROMPT_FILE = OUT_DIR / "prompt_templates.json"
CHECKSUM_FILE = OUT_DIR / "sha256sums.txt"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constraint_key(constraints: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(constraints.get("cuisine", "")).strip().lower(),
        int(constraints["max_calories"]),
        int(constraints["min_protein"]),
        (
            None
            if constraints.get("min_fiber") is None
            else int(constraints["min_fiber"])
        ),
        str(constraints.get("goal", "")).strip().lower(),
    )


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def main() -> None:
    for path in (DEV_FILE, BLIND_SIGNATURE_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    missing_adapter_files = [
        name
        for name in REQUIRED_ADAPTER_FILES
        if not (LORA_ADAPTER_DIR / name).exists()
    ]
    if missing_adapter_files:
        raise FileNotFoundError(
            "Missing frozen 100% LoRA adapter files: "
            + ", ".join(missing_adapter_files)
        )

    dev_rows = read_jsonl(DEV_FILE)
    blind_rows = read_jsonl(BLIND_SIGNATURE_FILE)

    if len(dev_rows) != 2069:
        raise RuntimeError(
            f"Expected 2069 development samples, got {len(dev_rows)}"
        )
    if len(blind_rows) != 250:
        raise RuntimeError(
            f"Expected 250 reserved blind signatures, got {len(blind_rows)}"
        )

    sample_ids = [str(row.get("sample_id", "")) for row in dev_rows]
    if not all(sample_ids):
        raise RuntimeError("A development row is missing sample_id.")
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate development sample_id detected.")

    dev_keys = {
        constraint_key(metadata_of(row)["constraints"])
        for row in dev_rows
    }
    blind_keys = {
        constraint_key(dict(row["constraints"]))
        for row in blind_rows
    }
    overlap = dev_keys & blind_keys
    if overlap:
        raise RuntimeError(
            f"Development/blind constraint overlap detected: {len(overlap)}"
        )

    common_system_prompt = (
        "You are the Fitness Home recommendation explanation model.\n"
        "The retrieval system has already selected the restaurant.\n"
        "Use only the information supplied in the current input.\n"
        "Do not recommend a different restaurant, invent facts, alter numbers, "
        "or infer unsupported health benefits.\n"
        "Clearly state unmet constraints.\n"
        "Return one concise evidence-grounded paragraph only."
    )

    prompts = {
        "common_system_prompt": common_system_prompt,
        "m1_template_rule": {
            "description": (
                "Deterministic explanation assembled from restaurant name, "
                "structured numerical evidence, and constraint_checks. "
                "No language model is used."
            ),
            "full": (
                "{restaurant_name} satisfies the stated requirements. "
                "{satisfied_sentences}"
            ),
            "weak": (
                "{restaurant_name} meets some of the stated requirements. "
                "{satisfied_sentences} However, {failed_sentences}"
            ),
            "partial": (
                "{restaurant_name} is a partial match. "
                "{satisfied_sentences} However, {failed_sentences}"
            ),
        },
        "m2_no_rag_user_prompt": (
            "User request:\n{query}\n\n"
            "Selected restaurant name:\n{restaurant_name}\n\n"
            "No restaurant facts or nutritional evidence are provided. "
            "Generate one concise recommendation explanation without inventing facts."
        ),
        "m3_m4_m5_evidence_prompt": (
            "Use the frozen development record's existing `input` field without "
            "adding or removing evidence. The same input is supplied to M3, M4, and M5."
        ),
    }

    methods = [
        {
            "method_id": "M1",
            "name": "Structured Rule/Template",
            "model": None,
            "rag_evidence": True,
            "lora": False,
            "role": "Non-LLM factual baseline",
            "input": (
                "Frozen structured evidence and constraint_checks from each "
                "development record."
            ),
            "important_note": (
                "This method may be strong on factual consistency but has limited "
                "linguistic flexibility. It is a deterministic baseline."
            ),
        },
        {
            "method_id": "M2",
            "name": "Base TinyLlama without RAG evidence",
            "model": BASE_MODEL,
            "rag_evidence": False,
            "lora": False,
            "role": "No-RAG ablation",
            "input": (
                "User request plus selected restaurant name only; category, tags, "
                "nutritional numbers, and constraint evaluation are withheld."
            ),
            "important_note": (
                "M2 tests whether a general small language model can remain factual "
                "when external evidence is absent."
            ),
        },
        {
            "method_id": "M3",
            "name": "Dense RAG + Base TinyLlama",
            "model": BASE_MODEL,
            "rag_evidence": True,
            "lora": False,
            "role": "RAG-only language-model baseline",
            "input": "Frozen full structured evidence from each development record.",
            "important_note": (
                "M2 versus M3 isolates the contribution of retrieved structured evidence."
            ),
        },
        {
            "method_id": "M4",
            "name": "Dense RAG + LoRA TinyLlama",
            "model": BASE_MODEL,
            "rag_evidence": True,
            "lora": True,
            "adapter_directory": str(LORA_ADAPTER_DIR),
            "role": "Full proposed text method",
            "input": "Identical frozen full evidence used by M3.",
            "important_note": (
                "M3 versus M4 isolates the contribution of LoRA domain adaptation."
            ),
        },
        {
            "method_id": "M5",
            "name": "Dense RAG + Llama-3.1-8B-Instruct",
            "model": TEACHER_MODEL,
            "rag_evidence": True,
            "lora": False,
            "role": "Teacher/reference upper bound",
            "input": "Identical frozen full evidence used by M3 and M4.",
            "important_note": (
                "The development references were originally generated by this teacher "
                "family. M5 must therefore be reported separately as a reference upper "
                "bound, not as an independent fair baseline for ROUGE."
            ),
        },
    ]

    protocol = {
        "protocol_version": "fitness_home_explanation_baseline_v1",
        "status": "frozen_before_baseline_execution",
        "research_questions": {
            "rq1": (
                "Does structured RAG evidence reduce factual and constraint errors "
                "relative to a base model without evidence?"
            ),
            "rq2": (
                "Does LoRA improve evidence interpretation and multi-constraint "
                "explanation quality beyond RAG alone?"
            ),
            "rq3": (
                "Can the 1.1B RAG+LoRA model approach the 8B teacher/reference model "
                "while using substantially fewer trainable parameters?"
            ),
            "rq4": (
                "How does the proposed neural explanation method compare with a "
                "deterministic structured template baseline?"
            ),
        },
        "benchmark": {
            "development_file": str(DEV_FILE),
            "development_samples": len(dev_rows),
            "development_unique_constraint_signatures": len(dev_keys),
            "development_sha256": sha256(DEV_FILE),
            "reserved_blind_file": str(BLIND_SIGNATURE_FILE),
            "reserved_blind_signatures": len(blind_rows),
            "reserved_blind_sha256": sha256(BLIND_SIGNATURE_FILE),
            "development_blind_constraint_overlap": 0,
            "blind_test_used": False,
            "important_note": (
                "The 2069 records are a Development Benchmark because they have already "
                "been used for method selection. The 250 blind signatures remain sealed."
            ),
        },
        "shared_generation_configuration": {
            "seed": 42,
            "max_input_tokens": 512,
            "max_new_tokens": 180,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "repetition_penalty": 1.0,
            "tinyllama_batch_size": 8,
            "llama_8b_batch_size": 4,
            "same_system_prompt_for_m2_to_m5": True,
        },
        "primary_metrics": [
            "All-Constraint Exact Accuracy",
            "Constraint-State Macro-F1",
            "Numeric Relation Accuracy",
            "Failed-Constraint Recall",
            "Restaurant Mention Accuracy",
            "Faithfulness",
            "Hallucination Rate",
            "Format Success",
            "ROUGE-L F1",
            "Token F1",
        ],
        "required_subgroup_metrics": [
            "Full",
            "Weak",
            "Partial",
            "Near-boundary numerical cases",
            "Cuisine mismatch",
            "Multiple failed constraints",
        ],
        "primary_comparisons": [
            "M2 vs M3: RAG contribution",
            "M3 vs M4: LoRA contribution",
            "M1 vs M4: deterministic factual baseline vs proposed method",
            "M4 vs M5: small adapted model vs teacher/reference upper bound",
        ],
        "reporting_rules": [
            "Report every frozen method, including negative results.",
            "Do not modify prompts or metrics after inspecting method outputs.",
            "Do not use the reserved final blind signatures during baseline development.",
            "Report M5 separately because teacher-generated references create ROUGE bias.",
            "Use the same 2069 development records for all five methods.",
        ],
        "planned_output_directory": str(
            HERE / "41_explanation_baseline_eval"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PROTOCOL_FILE.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    METHOD_FILE.write_text(
        json.dumps(
            {"methods": methods},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    PROMPT_FILE.write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output_files = (
        PROTOCOL_FILE,
        METHOD_FILE,
        PROMPT_FILE,
    )
    CHECKSUM_FILE.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in output_files
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("EXPLANATION BASELINE PROTOCOL FROZEN")
    print("=" * 72)
    print("Development samples       :", len(dev_rows))
    print("Development signatures    :", len(dev_keys))
    print("Reserved blind signatures :", len(blind_rows))
    print("Development/blind overlap :", 0)
    print("Main methods               : M1, M2, M3, M4")
    print("Reference upper bound      : M5")
    print("Frozen LoRA adapter        :", LORA_ADAPTER_DIR)
    print("Blind test used            : NO")
    print("Protocol                   :", PROTOCOL_FILE)
    print("Method manifest            :", METHOD_FILE)
    print("Prompt templates           :", PROMPT_FILE)
    print("Checksums                  :", CHECKSUM_FILE)


if __name__ == "__main__":
    main()
