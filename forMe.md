# ⚡ Multimodal Alpha-Signal Extractor

Ce projet est une plateforme de trading quantitatif de nouvelle génération qui fusionne l'**analyse technique visuelle** (via un Vision-Language Model fine-tuné) et l'**analyse de sentiment macro-économique** (via NLP) pour générer des signaux de trading structurés.

## 🏗️ Architecture (Clean Architecture & DDD)

La codebase a été restructurée pour suivre les principes du **Domain-Driven Design (DDD)** et de la **Clean Architecture (Architecture Hexagonale)**. Cela garantit une séparation stricte entre la logique métier pure et les dépendances externes (APIs, modèles LLM, UI).

### 1. Le Cœur Métier (`domain/`)
C'est le sanctuaire du projet. Il ne dépend de *rien d'autre*.
*   **`models.py`** : Définit les *Entities* et *Value Objects* en utilisant Pydantic. Les modèles comme `TradingSignal`, `SentimentResult`, et `TradingDecision` garantissent que la donnée est toujours valide et typée avant de circuler dans le système.
*   **`indicators.py`** : Contient les mathématiques pures (RSI, Bandes de Bollinger). Aucune notion de source de données n'existe ici.
*   **`services.py`** : Le `SignalMergerService` contient la logique métier complexe : "Que fait-on si le modèle visuel dit BUY mais que le sentiment NLP dit SELL ?".

### 2. L'Orchestration (`application/`)
C'est ici que les cas d'usage (Use Cases) sont définis.
*   **`ports.py`** : Définit des *Interfaces (Abstractions)*. Le Use Case dit "J'ai besoin d'un port pour récupérer des news (`NewsPort`) et d'un port pour lire une image (`VlmPort`)", mais il ne sait pas (et ne s'en soucie pas) de *comment* c'est implémenté (Ollama, vLLM, Yahoo...).
*   **`usecases.py`** : Le `AnalyzeMarketUseCase` orchestre le flux. Il récupère les données, génère l'image du graphique, puis lance **simultanément** (asynchrone) l'analyse visuelle et l'analyse de sentiment, avant de passer les résultats au `domain` pour la décision finale.

### 3. L'Infrastructure (`infrastructure/`)
C'est ici qu'on interagit avec le monde réel.
*   **`adapters/`** : C'est le code "sale" qui implémente les ports définis plus haut.
    *   `yfinance_adapter.py` : Se connecte à Yahoo Finance pour la data et les news.
    *   `mplfinance_adapter.py` : Transforme la data en image PNG.
    *   `llm_adapters.py` : Communique avec les modèles d'IA. Il gère les spécificités de `Ollama`, `vLLM` et surtout `Llama.cpp` (qui exécute notre modèle VLM fine-tuné). *Note : l'appel à llama.cpp étant synchrone par nature, il est encapsulé dans un `run_in_executor` pour ne pas bloquer l'Event Loop asynchrone du Use Case.*
*   **`logger.py`** : Système de log robuste (`loguru`), capable de cracher du JSON pour de la production (Datadog/ELK) ou des logs colorés pour le dev.

### 4. La Présentation (`presentation/`)
Les points d'entrée de l'application.
*   **`cli.py`** : L'interface en ligne de commande.
*   **`di_container.py`** : L'injecteur de dépendances. C'est lui qui lit le fichier `config.py` et qui branche les bons "Adapters" (ex: LlamaCpp) dans le "Use Case", avant de donner le Use Case prêt à l'emploi au CLI ou à Streamlit.
*   **`app/streamlit_app.py`** : Le dashboard interactif.

---

## 🤖 Le Modèle VLM Fine-Tuné

Contrairement aux approches classiques qui donnent des séries temporelles (chiffres) à un modèle, nous donnons une **image d'un graphique en chandeliers** (avec RSI et Bollinger) à un modèle de vision (VLM).

1.  **Génération du Dataset (`01_generate_dataset.py`)** : Le script crée des milliers d'images de graphiques boursiers passés, et les annote automatiquement (BUY/SELL/HOLD) en fonction de ce qui s'est réellement passé dans les jours suivants.
2.  **Fine-Tuning (`02_finetune...`)** : Nous utilisons **Unsloth (QLoRA)** pour fine-tuner un modèle *Qwen2.5-VL-3B*. Le modèle apprend à "voir" les patterns graphiques (ex: rebond sur une bande de Bollinger inférieure avec un RSI survendu).
3.  **Inférence (`Llama.cpp Adapter`)** : Le modèle fine-tuné est exporté au format GGUF. Plutôt que de dépendre d'un serveur lourd, le dashboard charge ce modèle GGUF directement via `llama.cpp`. L'UI force donc ce choix par défaut.

---

## 🚀 Flux d'exécution (Information Flow)

Quand vous cliquez sur "RUN ANALYSIS" dans le dashboard :

1.  Le `di_container` construit le pipeline.
2.  Le `YFinanceAdapter` récupère les prix et les news.
3.  Le Dashboard affiche les news (cliquables vers Yahoo Finance).
4.  Le `MplfinanceAdapter` dessine le graphique "en mémoire" et le sauvegarde.
5.  **Parallélisme** :
    *   Le `LlamaCppVlmAdapter` (notre Qwen fine-tuné) regarde l'image et donne son analyse technique.
    *   Le `OllamaSentimentAdapter` (LLaMA 3 local) lit les news et donne le sentiment macro.
6.  Le `SignalMergerService` (Domain) prend les deux signaux, applique les règles de gestion des risques, et sort la `TradingDecision` finale.
7.  Le Dashboard affiche la matrice d'exécution.
