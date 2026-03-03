# Multimodal Alpha-Signal Extractor

> **Any-to-Any** system that analyzes financial charts (Images) + market news (Text) to generate structured JSON trading signals — with a premium **Streamlit dashboard**.

```text
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE ARCHITECTURE                     │
│                                                             │
│  ┌──────────┐    ┌──────────────────────────────────┐      │
│  │ Chart    │───▶│ VLM (Vision-Language Model)      │      │
│  │ (Image)  │    │ ├── [Prod] llama3.2-vision (Ollama)      │
│  └──────────┘    │ └── [Custom] Qwen2.5-VL (llama_cpp)      │
│         │        └─────────────────┬────────────────┘      │
│         │                          │ VLM Signal JSON       │
│         │                          ▼                       │
│  ┌──────────┐    ┌──────────────────────────────────┐      │
│  │ News     │───▶│ LLM (Large Language Model)       │      │
│  │ (Text)   │    │ └── llama3:8b (Ollama)           │──┐   │
│  └──────────┘    └──────────────────────────────────┘  │   │
│                                                        │   │
│                        merge signals ◀─────────────────┘   │
│                              │                             │
│                      ┌───────▼────────┐                    │
│                      │ FINAL DECISION │                    │
│                      └────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Stack Technique

| Composant | Outil | Rôle |
|-----------|-------|------|
| Fine-tuning | **Unsloth** + QLoRA | Adapter Qwen2.5-VL-3B sur dataset multimodal (Google Colab T4) |
| Inférence VLM | **Ollama** (M4) / **llama-cpp** (Metal M4) | Servir le VLM localement via API ou GGUF direct |
| Orchestration | **LangChain** | Prompt multimodal, chaînage async, parsing Pydantic |
| Sentiment texte | **Ollama** | Extraction de sentiment via Llama3-8b local |
| Visualisation | **Plotly** + **mplfinance** | Charts interactifs (Streamlit) + statiques (dataset) |
| Interface | **Streamlit** | Dashboard premium avec dark mode |
| Tests | **pytest** | 31 tests unitaires (indicateurs, schemas, merger) |

## 🗂 Structure du Projet

```
multimodal-alpha-signal-extractor/
├── pyproject.toml                # Packaging moderne
├── requirements.txt              # Dépendances Python
├── .env.example                  # Variables d'environnement
├── config.py                     # Configuration centralisée
├── pytest.ini                    # Configuration tests
│
├── src/alpha_signal/             # 📦 Modules partagés
│   ├── __init__.py
│   ├── indicators.py             # RSI, Bollinger Bands
│   ├── schemas.py                # TradingSignal, SentimentResult, TradingDecision
│   ├── chart_renderer.py         # Mplfinance (PNG) + Plotly (interactif)
│   ├── data_fetcher.py           # Yahoo Finance (market data + news)
│   └── pipeline.py               # LangChain orchestrator (VLM + Sentiment)
│
├── app/                          # 🖥️ Interface Streamlit
│   └── streamlit_app.py          # Dashboard complet
│
├── tests/                        # ✅ Tests unitaires
│   ├── test_indicators.py        # Tests RSI, Bollinger, add_indicators
│   ├── test_schemas.py           # Tests Pydantic validation, roundtrip
│   └── test_merger.py            # Tests merge signals logic
│
├── 01_generate_dataset.py        # Étape 1 : Dataset synthétique
├── 02_finetune_vlm.py            # Étape 2a : Fine-tuning (CUDA)
├── 02_finetune_colab.py          # Étape 2b : Fine-tuning (Colab T4)
├── 03_serve_ollama.py            # Étape 3a : Serveur Ollama (M4)
├── 03_serve_vllm.py              # Étape 3b : Serveur vLLM (CUDA)
├── 04_langchain_pipeline.py      # Étape 4 : Pipeline CLI
├── 05_live_analysis.py           # Étape 5 : Analyse live CLI
│
├── dataset/                      # Données générées
└── models/                       # Checkpoints
```

## 🚀 Quick Start

### 1. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copier le fichier d'environment
cp .env.example .env

# Ollama — Pull les modèles nécessaires
ollama pull llama3.2-vision:11b
ollama pull llama3:8b
```

### 2. Lancer le Dashboard Streamlit ✨

```bash
streamlit run app/streamlit_app.py
```

L'interface se lance sur `http://localhost:8501` avec :

- 📊 **Charts interactifs Plotly** (candlestick + Bollinger + RSI)
- 📰 **News feed** en temps réel (Yahoo Finance)
- 🎯 **Pipeline complet** (VLM + Sentiment → Trading Decision)
- 🌙 **Dark mode** glassmorphism premium
- 📜 **Historique** des analyses

### 3. Pipeline CLI (alternatif)

```bash
# Mode démo (chart du dataset)
python 04_langchain_pipeline.py --demo

# Mode custom
python 04_langchain_pipeline.py \
    --image mon_chart.png \
    --news "Apple dépasse les attentes avec +12% de CA" \
    --output signal.json

# Analyse live
python 05_live_analysis.py --ticker NVDA --days 90
```

### 4. Tests

```bash
python -m pytest tests/ -v
# → 31 tests passés ✓
```

## ✅ Résultat d'Exécution (M4, 24 GB)

```json
{
  "final_action": "BUY",
  "final_confidence": 0.80,
  "vlm_signal": {
    "action": "BUY",
    "confidence": 0.8,
    "entry_price": 180.0,
    "stop_loss": 175.0,
    "take_profit": 190.0
  },
  "sentiment": {
    "sentiment": "BULLISH",
    "intensity": 0.8,
    "key_factors": ["record-breaking results", "12% revenue growth"]
  },
  "meta": {
    "vlm_model": "alpha-signal-q4km.gguf",
    "vlm_provider": "llama_cpp",
    "signals_aligned": true,
    "platform": "Apple Silicon M4"
  }
}
```

## ⚙️ Configuration

`config.py` — Toutes les configs en `dataclass` immuables.

`.env.example` — Variables d'environnement :

```bash
VLM_PROVIDER=llama_cpp           # ollama, llama_cpp, ou vllm
LLAMA_CPP_MODEL_PATH=~/Downloads/alpha-signal-q4km.gguf
OLLAMA_BASE_URL=http://localhost:11434
```

## 📄 License

MIT
