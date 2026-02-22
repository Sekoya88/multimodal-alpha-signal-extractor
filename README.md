# 🧠 Multimodal Alpha-Signal Extractor

> **Any-to-Any** system that analyzes financial charts (Images) + market news (Text) to generate structured JSON trading signals.

```
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE ARCHITECTURE                     │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Chart    │───▶│  Ollama VLM  │───▶│  VLM Trading     │  │
│  │ (Image)  │    │  (Llama3.2V) │    │  Signal (JSON)   │  │
│  └──────────┘    └──────────────┘    └────────┬─────────┘  │
│                                               │ merge      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ News     │───▶│  Ollama LLM  │───▶│  Final Trading   │  │
│  │ (Text)   │    │  (Llama3-8b) │    │  Decision (JSON) │  │
│  └──────────┘    └──────────────┘    └──────────────────┘  │
│                                                             │
│  ◀──────────── LangChain Orchestrator ──────────────▶       │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Stack Technique

| Composant | Outil | Rôle |
|-----------|-------|------|
| Fine-tuning | **Unsloth** + QLoRA | Adapter Qwen2.5-VL-3B sur dataset multimodal (Google Colab T4) |
| Inférence VLM | **Ollama** (M4) / **llama-cpp** (Metal M4) | Servir le VLM localement via API ou GGUF direct |
| Orchestration | **LangChain** | Prompt multimodal, chaînage async, parsing Pydantic |
| Sentiment texte | **Ollama** | Extraction de sentiment via Llama3-8b local |

## 🗂 Structure du Projet

```
multimodal-alpha-signal-extractor/
├── config.py                   # Configuration centralisée (dataclasses)
├── requirements.txt            # Dépendances Python
├── 01_generate_dataset.py      # Étape 1 : Génération du dataset synthétique
├── 02_finetune_vlm.py          # Étape 2a : Fine-tuning (CUDA — vLLM target)
├── 02_finetune_colab.py        # Étape 2b : Fine-tuning (Google Colab T4)
├── 03_serve_vllm.py            # Étape 3a : Serveur vLLM (CUDA)
├── 03_serve_ollama.py          # Étape 3b : Serveur Ollama (Apple Silicon M4)
├── 04_langchain_pipeline.py    # Étape 4 : Orchestrateur LangChain
├── decision.json               # Dernier output du pipeline
├── dataset/
│   ├── training_data.jsonl     # 84 samples multimodaux
│   └── demo_chart.png          # Chart de test extrait du dataset
└── models/                     # Checkpoints (après fine-tuning)
```

## 🚀 Quick Start (MacBook M4)

### 1. Prérequis

```bash
# Ollama (déjà installé si vous lisez ceci)
ollama pull llama3.2-vision:11b
ollama pull llama3:8b

# Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optionnel) Pour inférence GGUF directe avec le modèle fine-tuné sur Apple Silicon
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 2. Générer le Dataset

```bash
python 01_generate_dataset.py
# → 84 samples | 9.9 MB | AAPL 2024-2026
```

### 3. Fine-Tuner (Google Colab)

1. Upload `training_data.jsonl` sur Google Colab
2. `pip install unsloth` dans Colab (Runtime → T4 GPU)
3. `python 02_finetune_colab.py`
4. Télécharger le `.gguf` sur votre Mac
5. Exécuter le pipeline via `llama_cpp` (Metal) en configurant `config.py`

### 4. Exécuter le Pipeline

```bash
# Mode démo (utilise un chart du dataset)
python 04_langchain_pipeline.py --demo

# Mode custom
python 04_langchain_pipeline.py \
    --image mon_chart.png \
    --news "Apple dépasse les attentes avec +12% de CA" \
    --output signal.json
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
    "vlm_model": "llama3.2-vision:11b",
    "sentiment_model": "llama3:8b",
    "signals_aligned": true,
    "platform": "Apple Silicon M4"
  }
}
```

## ⚙️ Configuration

`config.py` — Toutes les configs en `dataclass` immuables :

| Config | Description |
|--------|-------------|
| `DatasetConfig` | Ticker, période, fenêtre, indicateurs |
| `TrainingConfig` | Modèle de base, LoRA rank, hyperparamètres |
| `VLLMConfig` | Host, port, GPU utilization (CUDA only) |
| `PipelineConfig` | `vlm_provider` (ollama/vllm), endpoints, retry |

Pour switcher entre Ollama et vLLM, modifiez `vlm_provider` dans `config.py` :

```python
vlm_provider: str = "ollama"  # ou "vllm" sur machine CUDA
```

## 📄 License

MIT
