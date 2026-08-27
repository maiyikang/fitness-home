# Fitness Home

Research code for **A Retrieval-Augmented and LoRA-Enhanced Framework for Multi-Constraint Restaurant Recommendation**.

## Overview

Fitness Home is a multi-constraint restaurant recommendation framework that combines structured retrieval, LoRA-based explanation generation, and optional visual understanding. It handles cuisine, nutrition, fitness-goal, budget, and related constraints while keeping generated explanations grounded in database evidence.

The structured knowledge base contains 4,996 restaurants and 137,352 menu-item records.

## Main Contributions

1. A unified RAG-LoRA framework that separates candidate retrieval from evidence-grounded recommendation-explanation generation.
2. A reduced-data study showing how much LoRA training data can be removed while retaining recommendation performance.
3. A modular VLM design with deterministic visual gating for handling aligned, noisy, and conflicting image evidence.
4. A benchmark-style evaluation framework containing multiple baselines, ablations, multi-seed experiments, and a final blind test on unseen constraint combinations.

## System Workflow

The final hybrid retriever combines BM25 top-50 retrieval, dense BGE top-50 retrieval, Reciprocal Rank Fusion with k=60, structured constraint-aware reranking, and final top-five selection.

The retrieved evidence is passed to a LoRA-adapted TinyLlama model for explanation generation. The optional modular VLM parses image information separately so that visual evidence cannot overwrite structured restaurant or nutritional facts.

## Models

- Embedding model: `BAAI/bge-small-en-v1.5`
- Teacher model: `meta-llama/Llama-3.1-8B-Instruct`
- Student model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Vision-language model: `meta-llama/Llama-3.2-11B-Vision-Instruct`

## Main Frozen Results

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
| MV1 Database Override Error Rate | 0% |

The final blind retrieval benchmark contains 1,000 natural-language queries generated from 250 unseen constraint signatures. The canonical end-to-end evaluation contains 250 cases.

## Repository Layout

- `app/`: Next.js application.
- `backend/`: FastAPI backend, structured data processing, retrieval, and model integration.
- `backend/rag/`: Dense retrieval and FAISS implementation.
- `backend/reproducibility/main20/`: Main-20K paper scripts, compact results, dependencies, and reproduction documentation.

## Reproduction

See the [Main-20K reproduction package](backend/reproducibility/main20/README.md).

The Git repository intentionally excludes raw datasets, generated Main-20K JSONL files, embeddings, FAISS indexes, checkpoints, LoRA adapters, and pretrained model weights. These assets are subject to size or third-party licensing restrictions.

## Data and Model Availability

The restaurant and menu data originate from public external datasets. Food-101 and pretrained Hugging Face models remain subject to their original licenses and access conditions. Access tokens and credentials must never be committed to this repository.
