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
