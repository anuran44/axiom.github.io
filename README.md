<div align="center">

# 🔥 Axiom: Neuro-Symbolic Inference Engine
**Zero Hallucinations. 100% Deterministic Mathematical Precision.**

[![Python](https://img.shields.io/badge/Python-3.10+-FF8C00?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Custom_Transformer-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![SymPy](https://img.shields.io/badge/SymPy-Verification_Kernel-013243?style=for-the-badge&logo=sympy&logoColor=white)](https://sympy.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 🌌 Overview

Standard Large Language Models (LLMs) rely on probabilistic token prediction, which frequently leads to arithmetic hallucinations when solving complex calculus. **Axiom** solves this flaw by implementing a **Neuro-Symbolic Dual-System Architecture**.

Constructed entirely from scratch without relying on pre-trained dependencies, Axiom combines the intuitive pattern-matching of a Neural Network with the flawless precision of a deterministic mathematical solver.

### 🧠 The Architecture

*   **System 1 (Neural Intuition):** A custom **Sequence-to-Sequence (Seq2Seq) PyTorch Transformer**. It processes human input, reads the mathematical structure through multi-head attention, and classifies the differential equation type.
*   **System 2 (Formal Verification):** A deterministic **SymPy Algebraic Kernel**. The neural pattern is routed here, where the system integrates the equation step-by-step and automatically back-propagates the result to mathematically prove its absolute accuracy.

---

## ✨ Features

*   **Zero-Hallucination Mathematics:** By decoupling pattern recognition from arithmetic execution, Axiom guarantees 100% accurate integration.
*   **Step-by-Step Inference Streaming:** Simulates biological thought processing, breaking down pattern identification, integration, and explicit solutions with controlled latency.
*   **Deep Amber UI:** A custom, glowing, glass-morphism web interface powered by Streamlit and dynamic CSS injection.
*   **Production-Ready Inference:** The model state dictionary (`.pt`) is locked and natively containerized for isolated, CPU-optimized deployment.

---

## 🚀 Quick Start (Docker)

The fastest way to experience Axiom locally is through its pre-packaged Docker container. This ensures you do not need to configure PyTorch or CUDA dependencies on your local machine.

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/axiom-engine.git](https://github.com/YOUR_USERNAME/axiom-engine.git)
cd axiom-engine

# 2. Build the Docker image
docker build -t axiom-engine .

# 3. Run the container
docker run -p 8501:8501 axiom-engine
