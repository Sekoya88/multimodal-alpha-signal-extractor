---
title: Alpha Signal Extractor
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app/streamlit_app.py
pinned: false
---

# Multimodal Alpha-Signal Extractor

> End-to-end system that fine-tunes a Vision-Language Model on candlestick charts,
> then combines visual technical analysis with NLP sentiment extraction to produce
> structured JSON trading signals — served through a premium Streamlit dashboard.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE ARCHITECTURE                         │
│                                                                    │
│  ┌──────────────┐       ┌─────────────────────────────────────┐   │
│  │ Candlestick  │──────▶│ Fine-tuned VLM (Qwen2.5-VL-3B)     │   │
│  │ Chart (PNG)  │       │  ├── llama.cpp  (Apple Silicon)     │   │
│  └──────────────┘       │  ├── Ollama     (llama3.2-vision)   │   │
│                         │  └── vLLM       (CUDA)              │   │
│                         └───────────────┬─────────────────────┘   │
│                                         │  TradingSignal JSON     │
│                                         ▼                         │
│  ┌──────────────┐       ┌─────────────────────────────────────┐   │
│  │ Financial    │──────▶│ Sentiment LLM (llama3:8b)           │   │
│  │ News (Text)  │       │  └── Ollama (text-only)             │──┐│
│  └──────────────┘       └─────────────────────────────────────┘  ││
│                                                                   ││
│                              merge_signals()  ◀──────────────────┘│
│                                    │                               │
│                            ┌───────▼────────┐                     │
│                            │ TradingDecision │                     │
│                            │  (Pydantic)     │                     │
│                            └────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Overview](#overview)
2. [Stack Technique](#stack-technique)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Reproducing the Fine-Tuned Model](#reproducing-the-fine-tuned-model)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Tests](#tests)
9. [Example Output](#example-output)
10. [License](#license)

---

## Overview

This project implements a **multimodal trading signal extractor** that reads financial charts
the same way a human trader would — visually — and merges that analysis with real-time
news sentiment. The pipeline is fully local (no cloud API calls), running on Apple Silicon
(M4) via `llama.cpp` or on CUDA machines via vLLM.

**Key capabilities:**

- **Visual technical analysis** — A fine-tuned Qwen2.5-VL-3B reads raw candlestick + Bollinger Bands + RSI charts
- **NLP sentiment scoring** — LLaMA 3 8B extracts bullish/bearish catalysts from Yahoo Finance news
- **Signal merging** — LangChain orchestrates both signals into a structured `TradingDecision` (action, confidence, entry/SL/TP)
- **Streamlit dashboard** — Premium Cyber-Fintech interface with glassmorphism, live charts, and real-time pipeline execution

---

## Stack Technique

| Component | Tool | Role |
|-----------|------|------|
| Fine-tuning | **Unsloth** + QLoRA | Adapt Qwen2.5-VL-3B on a synthetic multimodal dataset |
| VLM Inference | **llama.cpp** (Metal) / **Ollama** / **vLLM** | Serve the vision model locally |
| Orchestration | **LangChain** | Multimodal prompting, async chaining, Pydantic parsing |
| Sentiment | **Ollama** (llama3:8b) | Text-only sentiment extraction |
| Visualization | **Plotly** + **mplfinance** | Interactive charts (Streamlit) + static charts (dataset) |
| Interface | **Streamlit** | Dashboard with dark mode and glassmorphism |
| Tests | **pytest** | 31 unit tests (indicators, schemas, merger) |
| Config | **dataclasses** + `.env` | Centralized, immutable configuration |

---

## Project Structure

```
multimodal-alpha-signal-extractor/
├── pyproject.toml                # PEP 621 packaging + optional dependency groups
├── requirements.txt              # Flat dependency list
├── .env                          # Environment variables (git-ignored)
├── config.py                     # Centralized dataclass configs
├── pytest.ini                    # Test configuration
│
├── src/alpha_signal/             # Core application (Clean Architecture / DDD)
│   ├── domain/                   # Business rules (Models, Indicators, Services)
│   ├── application/              # Use Cases & Interface Ports
│   ├── infrastructure/           # Adapters (yfinance, LangChain, Llama.cpp) & Logging
│   └── presentation/             # CLI & DI Container
│
├── app/                          # Streamlit frontend
│   └── streamlit_app.py          # Cyber-Fintech dashboard
│
├── tests/                        # Unit tests (31 tests)
│   ├── test_indicators.py        # RSI, Bollinger, add_indicators
│   ├── test_schemas.py           # Pydantic validation, serialization roundtrip
│   └── test_merger.py            # Signal merge logic
│
├── 01_generate_dataset.py        # Step 1: Synthetic multimodal dataset
├── 02_finetune_vlm.py            # Step 2a: QLoRA fine-tuning (local CUDA)
├── 02_finetune_colab.py          # Step 2b: QLoRA fine-tuning (Google Colab T4)
├── 03_serve_ollama.py            # Step 3a: Ollama model management (Apple Silicon)
├── 03_serve_vllm.py              # Step 3b: vLLM serving (CUDA)
│
├── dataset/                      # Generated charts + JSONL (git-ignored)
└── models/                       # Checkpoints + GGUF files (git-ignored)
```

---

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running (for sentiment LLM)
- For VLM inference: one of the following backends configured
  - **llama.cpp** (recommended on Apple Silicon) — requires `llama-cpp-python`
  - **Ollama** — requires pulling `llama3.2-vision:11b`
  - **vLLM** — CUDA only

### Setup

```bash
# Clone and set up virtual environment
git clone https://github.com/<your-username>/multimodal-alpha-signal-extractor.git
cd multimodal-alpha-signal-extractor

python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull the sentiment model via Ollama
ollama pull llama3:8b

# (Optional) Pull the generic VLM if not using the fine-tuned GGUF
ollama pull llama3.2-vision:11b
```

### Environment Variables

Create a `.env` file at the project root (git-ignored by default):

```bash
# VLM backend: "llama_cpp", "ollama", or "vllm"
VLM_PROVIDER=llama_cpp

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VLM_MODEL=llama3.2-vision:11b
OLLAMA_TEXT_MODEL=llama3:8b

# llama.cpp — Fine-tuned GGUF model
LLAMA_CPP_MODEL_PATH=~/Downloads/alpha-signal-q4km.gguf
LLAMA_CPP_MMPROJ_PATH=models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf

# vLLM (CUDA only)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=alpha-signal-key
VLLM_MODEL_NAME=qwen2-vl-alpha-signal
```

---

## Reproducing the Fine-Tuned Model

The VLM used in this project is a **Qwen2.5-VL-3B** fine-tuned with QLoRA on a synthetic
multimodal dataset of candlestick charts + technical indicators. The full training pipeline
is reproducible in three steps.

### Step 1 — Generate the Synthetic Dataset

```bash
python 01_generate_dataset.py
```

This script:

- Downloads 2 years of AAPL daily data from Yahoo Finance
- Computes RSI (14) and Bollinger Bands (20, 2σ) on sliding 60-day windows
- Renders each window as a candlestick chart (mplfinance, PNG)
- Labels each sample (BUY/SELL/HOLD) based on forward returns + indicator thresholds
- Generates matching synthetic French-language financial news
- Outputs `dataset/training_data.jsonl` in Unsloth conversational VLM format

**Output:** `dataset/training_data.jsonl` + `dataset/charts/*.png`

### Step 2 — Fine-Tune the VLM

Two scripts are provided depending on your hardware:

| Script | Hardware | Base Model | Notes |
|--------|----------|------------|-------|
| `02_finetune_vlm.py` | Local CUDA (A100/H100, ≥16 GB VRAM) | `unsloth/Qwen2-VL-7B-Instruct` | Full-size model |
| `02_finetune_colab.py` | Google Colab T4 (free tier) | `unsloth/Qwen2.5-VL-3B-Instruct` | Memory-optimized for 15 GB VRAM |

```bash
# Local CUDA
pip install unsloth
python 02_finetune_vlm.py

# Google Colab
# Upload 02_finetune_colab.py + dataset/training_data.jsonl + dataset/charts/
# Run in a Colab notebook with T4 runtime
```

**Training configuration** (see `config.py` → `TrainingConfig`):

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Epochs | 3 |
| Batch size | 2 (× 4 gradient accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Quantization | QLoRA 4-bit (bnb) |
| Vision layers | Fine-tuned (cross-modal alignment) |
| Export | GGUF Q4_K_M |

**Output:** `models/qwen2-vl-alpha-signal/` (safetensors) + `alpha-signal-q4km.gguf`

### Step 3 — Serve the Model

**Apple Silicon (M4) — llama.cpp (recommended):**

No server needed. The pipeline loads the GGUF directly via `llama-cpp-python`:

```bash
# Set in .env
VLM_PROVIDER=llama_cpp
LLAMA_CPP_MODEL_PATH=~/Downloads/alpha-signal-q4km.gguf
```

You also need the multimodal projector file:

- Download `mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf` from the original Qwen2.5-VL GGUF repository
- Place it in `models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf`

**Apple Silicon — Ollama (generic model):**

```bash
python 03_serve_ollama.py --pull    # Pull required models
python 03_serve_ollama.py --check   # Verify availability
python 03_serve_ollama.py --test    # Test inference

# Set in .env
VLM_PROVIDER=ollama
```

**CUDA — vLLM:**

```bash
pip install vllm
python 03_serve_vllm.py

# Set in .env
VLM_PROVIDER=vllm
```

---

## Configuration

All configuration is centralized in `config.py` using frozen dataclasses:

| Dataclass | Purpose |
|-----------|---------|
| `DatasetConfig` | Ticker, window size, stride, indicator params, output paths |
| `TrainingConfig` | Base model, LoRA hyperparameters, training args, GGUF export |
| `VLLMConfig` | vLLM server settings (host, port, quantization, auth) |
| `PipelineConfig` | VLM backend selection, Ollama endpoints, retry policy, temperatures |

Runtime overrides are read from `.env` via `python-dotenv`.

---

## Usage

### Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501` with:

- **Top navigation** — Branding + system status
- **Command center** — Asset selector, time window, VLM engine, action buttons
- **Interactive Plotly chart** — Candlestick + Bollinger Bands + RSI + Volume
- **Real-time news feed** — Yahoo Finance articles
- **Execution Matrix** — Final BUY/SELL/HOLD decision with confidence, entry/SL/TP
- **Expandable panels** — Vision signal reasoning, NLP sentiment analysis, raw JSON

### CLI Pipeline

The pipeline is packaged via `pyproject.toml` and installs a global command `alpha-signal`.

```bash
# Live analysis (fetches real market data)
alpha-signal --ticker NVDA --days 90

# With JSON structured logs for production observability
alpha-signal --ticker AAPL --json-logs
```

---

## Tests

```bash
python -m pytest tests/ -v
```

31 tests covering:

- `test_indicators.py` — RSI computation, Bollinger Bands, edge cases, `add_indicators` integration
- `test_schemas.py` — Pydantic model validation, serialization roundtrip, boundary values
- `test_merger.py` — Signal alignment logic, confidence weighting, conflict resolution

---

## Example Output

Inference on Apple Silicon M4 (24 GB), using the fine-tuned GGUF via llama.cpp:

```json
{
  "final_action": "BUY",
  "final_confidence": 0.80,
  "vlm_signal": {
    "action": "BUY",
    "confidence": 0.8,
    "entry_price": 180.0,
    "stop_loss": 175.0,
    "take_profit": 190.0,
    "reasoning": "Price touching lower Bollinger Band with RSI at 32, suggesting oversold conditions. Candlestick pattern shows hammer formation on increased volume."
  },
  "sentiment": {
    "sentiment": "BULLISH",
    "intensity": 0.8,
    "key_factors": ["record-breaking results", "12% revenue growth"],
    "summary": "Strong earnings beat drives bullish consensus among analysts."
  },
  "meta": {
    "vlm_model": "alpha-signal-q4km.gguf",
    "vlm_provider": "llama_cpp",
    "signals_aligned": true,
    "platform": "Apple Silicon M4"
  }
}
```

---

## License

MIT
