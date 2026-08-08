# 🔥 Axiom: Neuro-Symbolic Inference Engine

> **Zero Hallucinations. 100% Deterministic Mathematical Precision.**

![Python](https://img.shields.io/badge/Python-3.10+-FF8C00?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom_Transformer-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-Verification_Kernel-013243?style=for-the-badge&logo=sympy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitLab CI](https://img.shields.io/badge/GitLab_CI-Pages-FC6D26?style=for-the-badge&logo=gitlab&logoColor=white)

---

## 🌌 Overview

Standard Large Language Models (LLMs) rely strictly on probabilistic token prediction. When tasked with calculus, integration, or differential equations, this architecture inevitably suffers from arithmetic hallucinations. 

**Axiom** solves this fundamental flaw by implementing a custom **Neuro-Symbolic Dual-System Architecture**. Constructed completely from scratch in PyTorch without relying on pre-trained API wrappers, Axiom pairs the intuitive pattern-recognition of a Neural Network with the absolute formal precision of a deterministic symbolic verification kernel.

---

## 🧠 System Architecture

Axiom operates using a two-system cognitive pipeline:

~~~text
                  ┌──────────────────────────────┐
                  │   Human Input (LaTeX/Text)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ SYSTEM 1: Neural Intuition (PyTorch Seq2Seq Transformer)         │
│  - Parses structural syntax via Multi-Head Self-Attention        │
│  - Classifies target mathematical operation and equation type    │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ SYSTEM 2: Symbolic Verification (SymPy Integration Kernel)       │
│  - Receives structured classification tokens                     │
│  - Computes step-by-step calculus deterministically              │
│  - Back-propagates step derivations and generates LaTeX output   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   100% Verified Response     │
                  └──────────────────────────────┘
~~~

---

## ✨ Key Features

* **Zero-Hallucination Calculus:** Decouples pattern recognition from symbolic manipulation for exact solutions.
* **Step-by-Step Reasoning Stream:** Visualizes the engine's internal thought process, from pattern detection to step-by-step algebraic manipulation.
* **Custom Glassmorphic Interface:** A deep-amber theme built with Streamlit and styled via dynamic CSS injection.
* **OpenAI-Compatible Microservice:** Includes a FastAPI deployment option (`/v1/chat/completions`) for local or cloud LLM workflows.
* **Containerized & CI/CD Ready:** Fully dockerized with automated deployment via GitLab CI/CD pipelines.

---

## 🚀 Quick Start

### Option 1: Run via Docker (Recommended)

No local Python or PyTorch setup required:

~~~bash
# 1. Clone the repository
git clone https://gitlab.com/YOUR_USERNAME/axiom-engine.git
cd axiom-engine

# 2. Build the Docker container
docker build -t axiom-engine .

# 3. Launch the container
docker run -p 8501:8501 axiom-engine
~~~

Open `http://localhost:8501` in your browser to view the interface.

---

### Option 2: Local Developer Setup

~~~bash
# 1. Clone and enter directory
git clone https://gitlab.com/YOUR_USERNAME/axiom-engine.git
cd axiom-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
~~~

---

### Option 3: Launch Local OpenAI-Compatible API

Serve the custom neural model as an API service:

~~~bash
# Launch FastAPI server on port 8000
uvicorn api_server:app --host 127.0.0.1 --port 8000
~~~

Query via `curl`:
~~~bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "axiom",
    "messages": [{"role": "user", "content": "dy/dx = x * y"}]
  }'
~~~

---

## 📂 Repository Structure

~~~text
axiom-engine/
├── .gitlab-ci.yml        # GitLab CI/CD Pipeline configuration
├── .streamlit/
│   └── config.toml       # Streamlit UI theme constraints
├── src/
│   ├── vocab.py          # Character & mathematical tokenizers
│   ├── model.py          # PyTorch Transformer architecture
│   ├── symbolic_engine.py# SymPy integration and LaTeX execution
│   └── pipeline.py       # Neuro-Symbolic router logic
├── weights/
│   ├── model.pth         # PyTorch neural weights state dictionary
│   └── vocab.json        # Character vocabulary mapping
├── app.py                # Main Streamlit web application
├── api_server.py         # FastAPI OpenAI-compatible endpoint
├── index.html            # GitLab Pages landing embed wrapper
├── Dockerfile            # Container deployment specification
└── requirements.txt      # Python dependencies
~~~

---

## 🧪 Demonstration Queries

Test Axiom's dual-system pipeline using these queries:

| Query Category | Example Input | Target Engine Response |
| :--- | :--- | :--- |
| **System Info** | `Tell me about yourself` | Neuro-symbolic architecture summary |
| **First-Order Separable** | `dy/dx = x * y` | $y(x) = C_1 e^{x^2/2}$ |
| **First-Order Linear** | `dy/dx + y = exp(x)` | $y(x) = \frac{e^x}{2} + C_1 e^{-x}$ |
| **Second-Order ODE** | `y'' + 4*y = x^2` | $y(x) = C_1 \sin(2x) + C_2 \cos(2x) + \frac{x^2}{4} - \frac{1}{8}$ |

---

## 🌐 Live Deployments

* **Interactive App:** [Axiom Live Interface](https://axiomappio.streamlit.app/)
* **GitLab Pages Portal:** [Axiom GitLab Portal](https://axiom-engine-1d445f.gitlab.io/)

---

<div align="center">
  <sub>Architected for precision. Engineered from scratch.</sub>
</div>