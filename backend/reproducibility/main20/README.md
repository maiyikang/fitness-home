# Main-20K Reproduction Package

This directory contains the release scripts and compact frozen results for the final Main-20K experiments reported in **A Retrieval-Augmented and LoRA-Enhanced Framework for Multi-Constraint Restaurant Recommendation**.

## Package Contents

- `scripts/`: 91 experiment and seed-specific Python scripts.
- `results/`: Compact JSON summaries, training metadata, and available SHA-256 records.
- `requirements-main20.txt`: Training, retrieval, and VLM dependencies.

The original 35 GB experiment workspace, raw data, JSONL datasets, predictions, indexes, checkpoints, adapters, and model weights are excluded.

## Experiment Stages

| Stage | Scripts | Purpose |
|---|---|---|
| Dataset construction | `step20`-`step23` | Query construction, teacher generation, Main-20K freezing, and splitting |
| Full-data LoRA | `step24`-`step25` | Full-data QLoRA training and evaluation |
| Reduced-data studies | `step26`-`step50` | Data budgets, selection strategies, ablations, and multi-seed tests |
| Retrieval and explanation | `step51`-`step55` | Explanation protocol, M1-M5, manual audit, B1-B6, and hybrid selection |
| VLM experiments | `step56`-`step58` | Monolithic VLM, modular VLM, and eight-set robustness testing |
| Final blind test | `step59` | Frozen retrieval and end-to-end evaluation |

## Data and Models

| Component | Resource |
|---|---|
| Knowledge base | 4,996 restaurants and 137,352 menu-item records |
| Full training split | 15,983 samples |
| Validation split | 1,948 samples |
| Development evaluation split | 2,069 samples |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Teacher model | `meta-llama/Llama-3.1-8B-Instruct` |
| Student model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| VLM | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| Vision data | Food-101 |

## Confirmed Full-Data Environment

| Item | Frozen value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 5.13.0 |
| PEFT | 0.19.1 |
| Accelerate | 1.14.0 |
| bitsandbytes | 0.49.2 |
| Datasets | 5.0.0 |
| GPU | NVIDIA GeForce RTX 4090 |

This environment record applies to the frozen full-data QLoRA run. The VLM summaries do not contain an independent complete environment snapshot.

## Training Configuration

| Parameter | Value |
|---|---:|
| Seed | 42 |
| Maximum sequence length | 512 |
| Epochs | 3 |
| Learning rate | 0.0002 |
| Training batch size | 1 |
| Evaluation batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Warm-up fraction | 0.03 |
| Maximum gradient norm | 0.3 |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | bfloat16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | All linear layers |

Under Transformers 5.x, the scripts pass `warmup_steps=0.03`; a floating-point value below 1 is interpreted as a fraction of the total training steps.

## Installation

Python 3.12 and a CUDA-capable Linux system are recommended.

1. Create a virtual environment with `python3.12 -m venv .venv`.
2. Activate it with `source .venv/bin/activate`.
3. Install the CUDA build with `python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128`.
4. Install the remaining packages with `python -m pip install -r backend/reproducibility/main20/requirements-main20.txt`.
5. Add the backend to the import path by setting `PYTHONPATH` to the repository's `backend` directory.

The exact original versions of NumPy, Pillow, sentence-transformers, FAISS, and torchvision were not recorded. Torchvision 0.23.0 is the official compatibility version for PyTorch 2.8.0.

## External Assets

A complete rerun requires:

- Restaurant and menu source data.
- Main-20K teacher-generated datasets and splits.
- Dense embeddings and a FAISS index.
- TinyLlama and Meta Llama model access.
- LoRA adapters when evaluating previously trained models.
- Food-101 for the VLM experiments.

Set `HF_HOME` to a writable Hugging Face cache directory. Set `FITNESS_HOME_FOOD101_ROOT` to the local Food-101 root. Never store access tokens in committed files.

## Running the Experiments

The scripts retain their original step numbers and experimental order. Run them from a separate ignored working directory after reviewing each script's expected inputs and command-line arguments.

There is no single-command full reproduction because large external inputs and licensed model artifacts are not distributed in this Git repository. Frozen summaries can be inspected directly without rerunning training.

## Frozen Results

| Evaluation | Result |
|---|---:|
| Development retrieval Full Match@1 | 85.47% |
| Final blind retrieval Full Match@1 | 84.40% |
| Final blind End-to-End Exact Accuracy | 76.80% |
| Final blind Faithfulness | 100% |
| Final blind Hallucination Rate | 0% |
| 10% data End-to-End Exact Accuracy | 76.53% ± 1.22% |
| MV1 Visual Cuisine Accuracy | 65.67% ± 2.13% |
| MV1 Conflict Resolution Exact Accuracy | 75.00% ± 2.75% |
| MV1 Multimodal Exact Accuracy | 31.08% ± 1.31% |
| MV1 Text Faithfulness | 99.00% |
| MV1 Database Override Error Rate | 0% |

## Provenance and Integrity

Files in `results/` preserve compact copies of the original summaries, metadata, and checksums. Absolute `/workspace/fitness-home/` paths inside frozen JSON files record original RunPod provenance and are not required installation paths.

Recorded full-data split hashes:

- Training: `08fdf5a5197aed3aaa0ec11b988e545097a22789164fab1460ee184bc0953820`
- Validation: `1dbdbe84ed1aa7887e61c7eddaddddde17366642a031c9deb9a869b165f80c47`
- Reserved evaluation: `b8cc6df07e30d057950ab24f72d7d1cef499aaf3cf9c2a2be6e34f26398e0c51`

## Limitations

Large datasets and model artifacts are not included. Exact pretrained-model revisions and several retrieval or image-processing package versions were not recorded. Exact bit-level reproduction may therefore depend on external assets and environment reconstruction.
