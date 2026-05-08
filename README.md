# NEXUS-ABSA: Adaptive Neuro-Symbolic Framework for Explainable Aspect-Based Sentiment Analysis 🚀

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![HuggingFace Accelerate](https://img.shields.io/badge/Accelerate-Multi--GPU-yellow.svg)
![Status](https://img.shields.io/badge/Status-Research_Ready-success.svg)

## 📌 Introduction
**NEXUS-ABSA** is a lightweight, explainable, and architecture-agnostic Neuro-Symbolic framework tailored for **Aspect-Based Sentiment Analysis (ABSA)**. This project investigates the architectural disparities between Encoders (e.g., DeBERTa, RoBERTa) and Decoders (e.g., Qwen2) in handling fine-grained sentiment dependencies. 

By leveraging the **Flexible TADA (Task-Aware Domain Adaptation)** strategy, NEXUS seamlessly injects symbolic knowledge (SenticNet / ConceptNet) into the neural backbone while updating **less than 20%** of the total parameters. This repository is specifically optimized for resource-constrained environments, featuring a full offline execution pipeline and multi-GPU distributed training via Hugging Face `Accelerate` on **Kaggle (2x T4 GPUs)**.

---

## 🌟 Key Features
1. **Parameter-Efficient Fine-Tuning (Flexible TADA):** Achieves SOTA-level accuracy by updating only the embedding layer and the last transformer block.
2. **Encoder vs. Decoder Benchmarking:** Rigorous empirical evaluation across bidirectional (DeBERTa/RoBERTa) and causal (Qwen2) architectures on the challenging **MAMS** dataset.
3. **Offline Knowledge Injection:** Eliminates API bottlenecks ($O(1)$ lookup) by pre-caching SenticNet and ConceptNet graphs for domain-specific vocabularies.
4. **Built-in XAI Engine:** Evaluates model transparency automatically using *Sufficiency* and *Infidelity* metrics dynamically applied to Aspect terms.
5. **Kaggle-Native MLOps Pipeline:** Specifically engineered to bypass the 12-hour session limits and internet bottlenecks on Kaggle using a Two-Phase (Colab -> Kaggle) execution strategy.

---

## 📂 Repository Structure (ساختار پروژه)

```text
NEXUS-ABSA/
│
├── 1_offline_prep/                # Phase 1: Data & Cache Preparation (Run on Colab)
│   ├── build_absa_caches.py       # Extracts aspect-adjacent vocab and queries SenticNet/ConceptNet
│   └── prep_kaggle_dataset.ipynb  # Downloads models & packages assets into a Kaggle-ready ZIP
│
├── 2_kaggle_workspace/            # Phase 2: Distributed Training (Run on Kaggle)
│   ├── nexus_trainer.ipynb        # Main Accelerate training loop (Dual T4 optimized)
│   └── modules/                   # Core architectural components
│       ├── __init__.py
│       ├── config.py              # Hyperparameters and experimental scenarios
│       ├── data_loader.py         # ABSA-specific dual-input ([CLS] Text [SEP] Aspect [SEP])
│       ├── knowledge_modules.py   # O(1) Offline knowledge graph readers
│       ├── model_architectures.py # NEXUS Base, TADA implementation, and Adapters
│       └── xai_engine.py          # Aspect-aware Explainability Metrics (Sufficiency/Infidelity)
│
├── requirements.txt               # Dependencies for both phases
└── README.md                      # This file
```

## ⚙️ Workflow & Execution Guide
This project requires a **Two-Phase Execution Strategy** to efficiently utilize Kaggle's free GPU quotas.

**Phase 1: Offline Asset Preparation (Google Colab / Local)**
Goal: Create a self-contained, offline dataset for Kaggle.

Install dependencies: pip install -r requirements.txt (including senticnet).

Run 1_offline_prep/build_absa_caches.py. This will download SemEval/MAMS datasets, extract relevant vocabulary, and generate absa_conceptnet_cache.json and absa_senticnet_cache.json.

Open and run 1_offline_prep/prep_kaggle_dataset.ipynb. It will download the heavy LLM weights (RoBERTa, DeBERTa, Qwen2) and package everything into nexus_absa_kaggle_input.zip.

Action: Upload this ZIP file to Kaggle as a Private Dataset named nexus_absa_kaggle_input.

**Phase 2: Distributed Training (Kaggle)**
Goal: Run experiments on 2x T4 GPUs.

Create a new Kaggle Notebook and enable **GPU T4 x2**.

Mount the dataset created in Phase 1 (Add Data -> Your Datasets).

Upload 2_kaggle_workspace/nexus_trainer.ipynb and the modules/ folder to the Kaggle working directory.

Run all cells in nexus_trainer.ipynb. The Hugging Face Accelerate library will automatically distribute the workload across both GPUs.

---