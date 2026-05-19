# QueryGPT: Retrieval-Augmented Conversational Generation Using a Domain-Specific Transformer Trained from Scratch

QueryGPT is a domain-focused conversational AI project that trains a GPT-style decoder model from scratch on technical Q&A text and augments inference with retrieval. The project is designed to compare pure language modeling against retrieval-augmented generation (RAG) for factual technical question answering.

## Project Objectives

- Train a transformer decoder from scratch on large-scale technical text.
- Add retrieval at inference time to ground responses in external context.
- Provide an interactive chatbot GUI for side-by-side RAG vs. no-RAG comparison.
- Support scalable experimentation with multi-device training and MLOps tooling.

## Text Sources

This project uses two publicly available Hugging Face datasets:

### 1) Primary Training Corpus
- **Dataset:** StackOverflow Posts by mikex86  
- **Link:** [https://huggingface.co/datasets/mikex86/stackoverflow-posts](https://huggingface.co/datasets/mikex86/stackoverflow-posts)  
- **Description:** ~60 million Markdown posts with metadata (post type, score, tags), predating Stack Overflow's 2024 LLM training restriction.  
- **Usage:** Core corpus for training context-response modeling.

### 2) Retrieval Index Corpus
- **Dataset:** Wikipedia English (`20220301.en` subset)  
- **Link:** [https://huggingface.co/datasets/wikipedia](https://huggingface.co/datasets/wikipedia)  
- **Usage:** Source text for FAISS retrieval index used during inference.

Both datasets are free, documented, and loadable via `datasets.load_dataset()`.

## Model Architecture

The base model is a GPT-style transformer decoder implemented with JAX + Flax:

- Causal self-attention decoder with learned positional embeddings (Flax NNX).
- Trained from scratch on StackOverflow Q&A context-response sequences.
- JIT-compiled training step using JAX for efficient execution.
- Multi-device data-parallel training via `jax.pmap` across available GPUs.

The base model is trained as a pure language model. Retrieval is introduced only at inference to enable a clean base-vs-RAG evaluation.

## Retrieval-Augmented Generation (RAG)

RAG is implemented as an inference-time augmentation pipeline:

1. Build a FAISS index over chunked passages from StackOverflow and Wikipedia.
2. Use a lightweight sentence encoder to embed corpus chunks and incoming user queries.
3. Retrieve top-3 relevant chunks for each query.
4. Prepend retrieved chunks to the decoder prompt as grounded context.
5. Generate responses with and without retrieval for comparison.

This setup supports quantitative evaluation of factual Q&A performance under:
- **Base model (no retrieval)**
- **RAG-enabled model (retrieval + generation)**

## Chatbot GUI

A Streamlit-based interface supports interactive testing:

- User enters a technical question.
- Retrieved context passages are displayed to the user.
- Model-generated answer is shown.
- Toggle allows side-by-side output comparison between RAG and no-RAG modes.

## MLOps and Experimentation

The project integrates core MLOps features:

- **Distributed training:** `jax.pmap` for multi-device execution.
- **Experiment tracking:** Weights & Biases (W&B) for metrics and run management.
- **Checkpointing:** Orbax checkpoint manager with resume support.

## Evaluation Focus

Primary evaluation compares base and RAG variants on factual technical QA quality, including:

- Groundedness to retrieved context
- Factual accuracy on technical prompts
- Response relevance and completeness

## Tech Stack

- **Modeling:** JAX, Flax (NNX)
- **Retrieval:** FAISS + sentence embeddings
- **Data:** Hugging Face Datasets
- **UI:** Streamlit
- **Tracking / Checkpoints:** W&B, Orbax

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Pipeline

Run the full retrieval data workflow:

```bash
# 1) Download and store raw datasets
python3 scripts/download_datasets.py

# 2) Build chunked retrieval corpus JSONL
python3 scripts/prepare_retrieval_corpus.py

# 3) Build FAISS index + metadata
python3 scripts/build_faiss_index.py
```

Or use Make targets:

```bash
make install
make data-download
make data-prepare
make index-build
```

## Status

This repository contains the implementation of a full retrieval-augmented conversational generation pipeline:

- Transformer pretraining from scratch on domain-specific text
- Retrieval indexing and RAG inference pipeline
- Interactive demo interface
- Training and experiment infrastructure for reproducibility

