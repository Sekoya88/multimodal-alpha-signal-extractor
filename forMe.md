# ⚡ Multimodal Alpha-Signal Extractor — forMe (Interview Prep)

> **One-liner** : Plateforme quant de nouvelle génération fusionnant un **VLM fine-tuné (Qwen2.5-VL-3B)** pour l'analyse visuelle graphique + un **LLM sentiment** (LLaMA 3) sur les actualités macro, avec un pipeline d'alignement RLHF complet (SFT → DPO → GRPO).
>
> **Notes détaillées** → `obsidian/AI-Vision/`

---

## 🏗️ Architecture — Clean Architecture & DDD

La codebase suit le **Domain-Driven Design** et la **Clean Architecture (Hexagonale)**.

### Couches

- **`domain/`** : Zéro dépendance externe. Contient les modèles Pydantic (`TradingSignal`, `SentimentResult`, `TradingDecision`), les calculs purs d'indicateurs (RSI, Bollinger, MACD), et le `SignalMergerService`.
- **`application/`** : Ports (interfaces ABC) + Use Cases. `AnalyzeMarketUseCase` lance **en parallèle async** (`asyncio.gather`) l'analyse VLM et le sentiment NLP.
- **`infrastructure/`** : Adapters concrets — `YFinanceAdapter`, `MplfinanceAdapter`, `LlamaCppAdapter`, `InferenceCacheAdapter` (Redis).
- **`presentation/`** : DI Container, CLI Typer, Streamlit Dashboard.

### Règle de Fusion des Signaux (`SignalMergerService`)

```
VLM dit BUY + NLP dit BULLISH  → confidence = VLM*0.7 + sentiment_intensity*0.3
VLM dit BUY + NLP dit BEARISH  → confidence = VLM*0.5 (réduction — conflit)
VLM dit HOLD                   → HOLD toujours, confidence = VLM*0.8
```

---

## 🤖 Pipeline de Fine-Tuning (5 Sprints)

### Sprint 0 — Génération Dataset Auto-Annoté (`01_generate_dataset.py`)

- Fenêtres glissantes 60 jours (`stride=5`) sur données OHLCV yFinance
- Oracle mathématique : `forward_return_5j > 0.5% → BUY`, `< -0.5% → SELL`, sinon `HOLD`
- Graphiques candlestick (mplfinance) avec RSI + Bollinger → encodés en base64 dans un JSONL
- Format : messages multimodaux compatible Qwen chat template (image + texte dans le même JSON)

### Sprint 1 — SFT (Supervised Fine-Tuning) QLoRA (`02_finetune_colab.py`)

- **Unsloth FastVisionModel** : 2× plus rapide grâce à Flash Attention 2 + memory efficient cross-attention
- **BitsAndBytes NF4 4-bit** : modèle 3B → ~3 Go VRAM (vs 12 Go en bf16)
- **QLoRA r=16, alpha=16** : seulement ~2-5% des paramètres entraînés (matrices B×A de bas rang greffées sur chaque couche linéaire)
- **Gradient Checkpointing** : recalcule les activations en backward au lieu de les stocker → -30% VRAM
- Export **GGUF Q4_K_M** pour llama.cpp sur Apple Silicon

### Sprint 2 — DPO Alignment (`04_dpo_alignment.py`)

- **Direct Preference Optimization** : paires `(chosen oracle, rejected prediction)` sans RM externe
- Génère les paires : model inference → compare à oracle → si faux = rejected, si juste = chosen + synthétique faux
- Loss DPO : `σ(β * (log π(y_w|x)/π_ref - log π(y_l|x)/π_ref))` avec `β=0.1`
- **Piège** : ne jamais appeler `.to(device)` sur un modèle BNB 4-bit → OOM killer silencieux

### Sprint 3 — Reward Model Visuel (`05_reward_model.py`)

- Backbone Qwen2.5-VL **gelé** + tête MLP-2-couches entraînable → score `[0,1]`
- Training data : oracle correct → reward 1.0, oracle flippé → reward 0.0 (50/50 balancé)
- Pluggé comme **6ème nœud** dans le pipeline après le `SignalMergerService`

### Sprint 4 — GRPO (`06_grpo_training.py`)

- **Group Relative Policy Optimization** (DeepSeek) — pas de RM, pas de paires
- Génère N=8 prédictions par graphique avec `temperature=0.7` → diversité intra-groupe
- Récompense composite : `r = 0.6*directional_accuracy + 0.4*calibration`
- Normalisation intra-groupe → avantage relatif → PPO clipping `ε=0.2`
- **Learning rate très petit** : `1e-5` (modèle déjà aligné par DPO)

### Sprint 5 — Extension Temporelle Multi-Frame (`07_temporal_extension.py`)

- N=8 graphiques consécutifs de 60 jours (`stride=5`) passés en multi-image dans un seul prompt
- Coverage : `60 + 7×5 = 95 jours` (+58% de contexte)
- **Positional Embeddings temporels appris** pour que le modèle connaisse l'ordre des frames
- Context window montée à `max_seq_length=4096` pour absorber tous les tokens visuels

### Sprint 6 — Production vLLM + Cache Redis (`08_inference_benchmark.py`)

- **vLLM** : PagedAttention + Continuous Batching → 64 requêtes simultanées
- **Speculative Decoding** : Qwen 0.5B brouillon → Qwen 3B verifier → gain 2-3× latence
- **Redis TTL 1h** : hash MD5(image+prompt) → cache hit en < 5ms vs ~400ms inference
- **Benchmark async** : `asyncio.gather` → mesure p50/p95/p99 à 1/4/16/32 utilisateurs concurrents

---

## 🛠️ Multi-Backend Inférence (`PipelineConfig`)

```
vlm_provider = "llama_cpp"  → Apple Silicon M4 (Metal GPU, GGUF)
vlm_provider = "vllm"       → NVIDIA CUDA (A100, T4...)
vlm_provider = "ollama"     → Développement rapide (llama3.2-vision)
```

Le changement de backend ne nécessite **aucune modification de code métier** — seul `config.py` change.

---

## � Bugs & Debugging Mémorisés

| Symptôme | Cause | Fix |
|---------|-------|-----|
| Script Colab ✅ après 1 min mais sans output | OOM Killer Linux tue le process silencieusement | Supprimer `.to(device)` sur BNB 4-bit |
| `CUDA required` après reset Colab | GPU désactivé par défaut au reset | Runtime → Change Runtime Type → T4 |
| `zip error: Nothing to do` | `models/dpo-adapter` non créé (crash avant) | Regarder les logs : crash pendant training |
| `zip warning: name not matched` | Path relatif dans `!zip` depuis Colab | Utiliser path absolu `/content/` |
| Images trop grandes → OOM | AutoProcessor sans borne de pixels | `max_pixels=512*28*28` |

---

## 🔑 Mots-Clés par Domaine

**Architecture** : Clean Architecture, DDD, Hexagonale, Ports & Adapters, DI Container, asyncio.gather  
**VLM** : ViT patches 28×28, AutoProcessor, apply_chat_template, multi-image, positional embeddings  
**Fine-Tuning** : QLoRA, NF4, Gradient Checkpointing, Flash Attention 2, LoRA r=16  
**Alignement** : DPO β=0.1, GRPO group_size=8, PPO ε=0.2, calibration reward  
**Production** : vLLM, PagedAttention, Speculative Decoding γ=4, Redis TTL, p95 latency  
**Finance** : Forward Return Oracle, Directional Accuracy, Calibration, Bollinger, RSI, MACD  
