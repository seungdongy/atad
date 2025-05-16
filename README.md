# ATAD: Agent-Centric Textual Anomaly Detection benchmark protocol

This repository contains the official implementation of **ATAD**, a benchmark developed by **LG AI Research** and submitted to **NeurIPS 2025 (Datasets & Benchmarks Track)**.

---

## 🔍 Overview

Traditional evaluation of large language models (LLMs) relies on static datasets, which are limited in scalability and fail to capture evolving reasoning abilities.

**ATAD** introduces a dynamic, agent-centric protocol where:
- A **Teacher** agent generates problems,
- An **Orchestrator** validates them and ensures coherence/fairness,
- A **Student** agent attempts to solve them.

If the student succeeds, the orchestrator requests a harder version; if not, the problem is finalized. This process enables **difficulty scaling** without human curation.

The benchmark uses **text anomaly detection** tasks requiring logical reasoning across multiple sentences, revealing weaknesses missed by traditional benchmarks.

---

## 📁 Repository Structure

```bash
.
├── generation/            # Benchmark generation code
├── evaluation/            # Benchmark evaluation code
├── requirements.txt       # Required Python packages
├── LICENSE                # MIT License
└── README.md              # This file
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run example benchmark generation
```bash
python generation/orchestrator_agentic_generator.py --config config.yaml
```
### 3. Evaluate model output
```bash
python evaluation/eval_agentic_models.py --config eval_config.yaml --dataset 'path/your/data'
```
---
## 📦 Dataset

The full dataset and Croissant metadata will be made available via the [Hugging Face Hub](https://huggingface.co/datasets/ADsquad/ATAD).

We follow the NeurIPS 2025 dataset hosting guidelines, including:
- Validated [Croissant metadata](https://github.com/mlcommons/croissant)
- Publicly accessible storage
- Consistent versioning and permanence

A direct link will be updated here upon acceptance.


https://huggingface.co/datasets/ADsquad/ATAD


