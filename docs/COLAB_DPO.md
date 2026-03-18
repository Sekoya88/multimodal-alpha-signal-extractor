# DPO Alignment sur Google Colab — Procédure reproductible

Cette procédure évite les conflits torchao/torch, Unsloth RecursionError et les erreurs d'import.

## Prérequis

- **Runtime** : Colab avec GPU T4 (gratuit) ou A100 (Colab Pro)
- **Runtime** : Disconnect + Delete runtime avant de commencer (environnement propre)

---

## Procédure en 4 cellules

### Cellule 1 — Clone et setup SANS Unsloth (chemin stable)

```python
# Clone
!rm -rf multimodal-alpha-signal-extractor 2>/dev/null
!git clone -b feature/dpo-grpo-extensions https://github.com/Sekoya88/multimodal-alpha-signal-extractor.git
%cd multimodal-alpha-signal-extractor

# Désinstaller torchao AVANT tout (incompatible torch 2.5 Colab gratuit)
!pip uninstall torchao -y 2>/dev/null || true

# Install DPO deps SANS unsloth (évite RecursionError + torch.int1)
!pip install -e ".[dpo-colab]" -q

# Si torch 2.5 : garder transformers/trl fournis par dpo-colab
# Si erreur MODEL_FOR_VISION_2_SEQ : le shim dans dpo_alignment_service patche auto
```

### Cellule 2 — Générer les données (si pas déjà fait)

```python
!python 01_generate_dataset.py
```

### Cellule 3 — Builder les pairs DPO (1ère fois uniquement)

```python
!PYTHONPATH=src python 04_dpo_alignment.py --max-samples 50
# (Sans --skip-pairs pour générer dpo_pairs.jsonl)
```

### Cellule 4 — Patch du format image + entraînement

```python
# Patch prompt pour insert <|image_pad|>
import json
path = "/content/multimodal-alpha-signal-extractor/dataset/dpo_pairs.jsonl"
with open(path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]
for item in items:
    msg = item["prompt"][-1]
    if isinstance(msg.get("content"), str):
        msg["content"] = [{"type": "image"}, {"type": "text", "text": msg["content"]}]
with open(path, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("✅ Patch done")

# Lancer DPO
!cd /content/multimodal-alpha-signal-extractor && DPO_USE_UNSLOTH=0 PYTHONPATH=src python 04_dpo_alignment.py --max-samples 50 --skip-pairs
```

---

## Si Colab a torch 2.10 (Pro / runtime récent)

Avec torch 2.10, tu peux utiliser Unsloth :

```python
!pip install ".[train]" -q
!DPO_USE_UNSLOTH=1 PYTHONPATH=src python 04_dpo_alignment.py --max-samples 50 --skip-pairs
```

(Après le patch du dataset en Cellule 4.)

---

## Dépannage

| Erreur | Action |
|--------|--------|
| `torch.int1` | `!pip uninstall torchao -y` |
| `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES` | `!pip install "transformers>=4.45" "trl>=0.11,<0.13" -q` |
| `RecursionError` | Utiliser `DPO_USE_UNSLOTH=0` |
| `tokens: 0, features: 1820` | Exécuter le patch (Cellule 4) avant `--skip-pairs` |
