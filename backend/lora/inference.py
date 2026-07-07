from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import MAX_NEW_TOKENS, TEMPERATURE
from core.paths import PROJECT_ROOT


BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_MODEL_DIR = PROJECT_ROOT / "models" / "fitness_home_lora_v1"


class LoraInference:
    def __init__(self) -> None:
        self.device = self.detect_device()

        print("=" * 60)
        print("Fitness Home LoRA Inference")
        print("=" * 60)
        print(f"Device : {self.device}")

        self.tokenizer = self.load_tokenizer()

        self.model = self.load_model()

        self.model.eval()

        print("Inference model loaded successfully.")
        print("=" * 60)

    def detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def load_tokenizer(self) -> AutoTokenizer:
        print("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        return tokenizer

    def load_base_model(self):
        print("Loading TinyLlama base model...")

        if self.device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float32,
            )

            model.to(self.device)

        return model

    def load_model(self):
        model = self.load_base_model()

        if LORA_MODEL_DIR.exists():
            print("Loading LoRA adapter...")

            model = PeftModel.from_pretrained(
                model,
                str(LORA_MODEL_DIR),
            )

            print("LoRA adapter loaded.")

        else:
            print("LoRA adapter not found.")
            print("Using TinyLlama base model.")

        return model

    def build_messages(
        self,
        prompt: str,
    ):
        return [
            {
                "role": "system",
                "content": (
                    "You are the Fitness Home recommendation assistant.\n"
                    "You must explain recommendations only.\n"
                    "Never change the retrieved restaurant.\n"
                    "Never hallucinate restaurant names.\n"
                    "Never invent nutrition facts.\n"
                    "Only explain why the selected restaurant satisfies the user's request."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def generate(
        self,
        prompt: str,
    ) -> str:

        messages = self.build_messages(prompt)

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                repetition_penalty=1.10,
                top_p=0.90,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return self.clean_response(response)

    def clean_response(
        self,
        response: str,
    ) -> str:

        response = response.strip()

        unwanted_prefix = [
            "<|assistant|>",
            "</s>",
            "<s>",
        ]

        for prefix in unwanted_prefix:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        return response


def main() -> None:

    inference = LoraInference()

    prompt = """
User Query:
I want a high-protein Japanese meal under 600 calories.

Retrieved Restaurant:
Samurai

Retrieved Menu:
Hibachi Chicken

Evidence:

Calories:
350 kcal

Protein:
38 g

Fiber:
9 g

Instruction:

Explain why this restaurant matches the user's request.

Do NOT recommend another restaurant.

Do NOT invent nutrition facts.
""".strip()

    print()
    print("=" * 60)
    print("Generating...")
    print("=" * 60)

    response = inference.generate(prompt)

    print()
    print("=" * 60)
    print("Model Response")
    print("=" * 60)

    print(response)

    print("=" * 60)


if __name__ == "__main__":
    main()