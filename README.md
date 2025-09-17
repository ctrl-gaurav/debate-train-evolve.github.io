# Debate, Train, Evolve: Self-Evolution of Language Model Reasoning

[![EMNLP 2025](https://img.shields.io/badge/EMNLP%202025-Main%20Conference-brightgreen)](https://2025.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.15734-b31b1b.svg)](https://arxiv.org/abs/2505.15734)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-green)](https://github.com/ctrl-gaurav/debate-train-evolve)
[![Models](https://img.shields.io/badge/Models-🤗%20HuggingFace-yellow)](https://huggingface.co/)

> 🎉 **[EMNLP 2025 Main Conference]** - We are excited to announce that our paper has been **accepted to EMNLP 2025 Main Conference**!

This repository contains the official implementation and project page for **"Debate, Train, Evolve: Self-Evolution of Language Model Reasoning"**, accepted at **EMNLP 2025 Main Conference**.

## 🎯 Overview

We introduce **DTE (Debate, Train, Evolve)**, a novel ground truth-free training framework that uses multi-agent debate traces to evolve a single language model's reasoning capabilities. Our approach combines the benefits of multi-agent debate with the efficiency of single-model inference.

### Key Contributions

- **🔄 RCR Prompting Strategy**: Reflect-Critique-Refine prompting that reduces sycophancy by 50%
- **⚡ DTE Framework**: Self-evolution without external supervision using multi-agent debate traces
- **📈 Strong Performance**: 8.92% average accuracy gain on GSM-PLUS dataset
- **🌐 Cross-Domain Generalization**: 5.8% improvement on unseen benchmarks

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ctrl-gaurav/debate-train-evolve.git
cd debate-train-evolve

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from dte import DTEFramework

# Initialize the DTE framework
dte = DTEFramework(
    base_model="qwen2.5-1.5b",
    num_agents=3,
    max_rounds=3
)

# Run multi-agent debate and evolve the model
evolved_model = dte.evolve(
    dataset="gsm8k",
    epochs=1,
    batch_size=32
)

# Use the evolved model for inference
result = evolved_model.generate("What is 15% of 240?")
```

## 📊 Results

### Main Performance Results

| Model | GSM8K | GSM-Plus | ARC-Challenge | Best Improvement |
|-------|-------|----------|---------------|------------------|
| **Qwen-2.5-1.5B** | 62.77 → 73.09 | 42.00 → 55.92 | 69.21 → 68.36 | **+13.92%** (GSM-Plus) |
| **Qwen-2.5-3B** | 84.08 → 86.05 | 61.75 → 69.50 | 83.53 → 83.95 | **+7.75%** (GSM-Plus) |
| **Llama-3.1-8B** | 81.73 → 86.81 | 55.62 → 66.17 | 77.65 → 86.53 | **+10.55%** (GSM-Plus) |

### Cross-Domain Generalization

Our evolved models show strong generalization across different reasoning tasks:
- **Mathematical Reasoning**: Average +8.92% on GSM-PLUS
- **Science Reasoning**: Average +3.67% on ARC-Challenge
- **Commonsense Reasoning**: Average +2.1% on CommonsenseQA

## 🔬 Framework Details

### DTE Pipeline

1. **Debate Phase**: Multiple agents engage in structured reasoning debates using RCR prompting
2. **Training Phase**: Extract high-quality reasoning traces and train with GRPO
3. **Evolution Phase**: Replace the original model with the evolved version

### RCR Prompting Strategy

Our Reflect-Critique-Refine approach structures agent responses through three phases:
- **Reflect**: Self-critique of current reasoning
- **Critique**: Evaluation of peer rationales
- **Refine**: Updated response with novel reasoning

## 📁 Repository Structure

```
debate-train-evolve/
├── src/
│   ├── agents/          # Multi-agent debate implementation
│   ├── training/        # GRPO training utilities
│   ├── prompts/         # RCR prompting strategies
│   └── evaluation/      # Benchmark evaluation scripts
├── configs/             # Configuration files
├── data/               # Dataset preprocessing
├── scripts/            # Training and evaluation scripts
├── models/             # Pre-trained and evolved models
└── results/            # Experimental results
```

## 🔧 Evaluation

### Benchmarks Supported

- **GSM8K**: Grade school math problems
- **GSM-Plus**: Adversarial mathematical reasoning
- **ARC-Easy/Challenge**: Science reasoning
- **CommonsenseQA**: Commonsense reasoning

### Running Evaluation

```bash
# Evaluate on GSM8K
python scripts/evaluate.py \
    --model qwen2.5-1.5b-evolved \
    --dataset gsm8k \
    --batch_size 16

# Cross-domain evaluation
python scripts/cross_domain_eval.py \
    --source_dataset gsm8k \
    --target_datasets arc_challenge,commonsenseqa
```

## 📄 Citation

If you find our work useful, please cite:

```bibtex
@article{srivastava2025debate,
  title={Debate, Train, Evolve: Self-Evolution of Language Model Reasoning},
  author={Srivastava, Gaurav and Bi, Zhenyu and Lu, Meng and Wang, Xuan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  note={To appear}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was supported by NSF NAIRR Pilot with PSC Neocortex, NCSA Delta; Amazon, Cisco Research, Commonwealth Cyber Initiative, Amazon–Virginia Tech Center for Efficient and Robust Machine Learning, and Sanghani Center for AI and Data Analytics at Virginia Tech.

## 📞 Contact

- **Gaurav Srivastava**: [gks@vt.edu](mailto:gks@vt.edu)
- **Xuan Wang** (Corresponding Author): [xuanw@vt.edu](mailto:xuanw@vt.edu)

---

**Project Page**: [https://debate-train-evolve.github.io](https://debate-train-evolve.github.io)