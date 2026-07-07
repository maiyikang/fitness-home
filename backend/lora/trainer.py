from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from core.paths import DATA_DIR, PROJECT_ROOT


BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TRAINING_DATA_FILE = DATA_DIR / "lora_training_data.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "fitness_home_lora_v1"

MAX_LENGTH = 512
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


class FitnessHomeDataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.samples = self.load_samples(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_samples(self, data_file: Path) -> List[Dict[str, str]]:
        if not data_file.exists():
            raise FileNotFoundError(f"Training data not found: {data_file}")

        samples: List[Dict[str, str]] = []

        with data_file.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    samples.append(json.loads(line))

        if len(samples) == 0:
            raise ValueError("Training dataset is empty.")

        return samples

    def format_sample(self, sample: Dict[str, str]) -> str:
        instruction = sample.get("instruction", "")
        user_input = sample.get("input", "")
        output = sample.get("output", "")

        return (
            "<|system|>\n"
            "You are a fitness meal recommendation assistant. "
            "You generate evidence-based restaurant and menu explanations.\n"
            "</s>\n"
            "<|user|>\n"
            f"Instruction:\n{instruction}\n\n"
            f"User Request:\n{user_input}\n"
            "</s>\n"
            "<|assistant|>\n"
            f"{output}\n"
            "</s>"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.format_sample(self.samples[index])

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_tokenizer() -> AutoTokenizer:
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model() -> AutoModelForCausalLM:
    print("Loading base model...")

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("CUDA detected. Loading model with 4-bit QLoRA configuration.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model)

    else:
        print("CUDA not detected. Loading model without 4-bit quantization.")
        print("This is only suitable for a small local test. Formal training should use a GPU server.")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=None,
        )

    return model


def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    print("Applying LoRA configuration...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    print("Trainable parameters:")
    model.print_trainable_parameters()

    return model


def train() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_model()
    model = apply_lora(model)

    dataset = FitnessHomeDataset(
        data_file=TRAINING_DATA_FILE,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    print(f"Loaded training samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting LoRA training...")
    trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("LoRA training completed.")
    print(f"Adapter saved to: {OUTPUT_DIR}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()