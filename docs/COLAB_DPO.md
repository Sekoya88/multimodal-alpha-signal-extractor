# DPO Alignment sur Google Colab — Procédure reproductible

Runtime : GPU T4 (gratuit). À chaque nouvelle session, exécuter les cellules dans l'ordre.

---

## Cellule 1 — Reset + clone propre

```python
import os
os.chdir("/content")

!rm -rf /content/multimodal-alpha-signal-extractor
!git clone -b feature/dpo-grpo-extensions \
    https://github.com/Sekoya88/multimodal-alpha-signal-extractor \
    /content/multimodal-alpha-signal-extractor

!ls /content/multimodal-alpha-signal-extractor/04_dpo_alignment.py
```

## Cellule 2 — Install dépendances

```python
!pip install -e "/content/multimodal-alpha-signal-extractor" -q
!pip install trl transformers peft bitsandbytes accelerate wandb -q
```

## Cellule 3 — Patch config.py

```python
config_path = "/content/multimodal-alpha-signal-extractor/config.py"
with open(config_path, "r") as f:
    content = f.read()

bad_block = """# ============================================================================
# Singleton instances (import directly)
# ============================================================================
dataset_cfg = DatasetConfig()
training_cfg = TrainingConfig()
dpo_cfg = DPOConfig()
reward_model_cfg = RewardModelConfig()
grpo_cfg = GRPOConfig()
temporal_cfg = TemporalConfig()
benchmark_cfg = BenchmarkConfig()
vllm_cfg = VLLMConfig()
pipeline_cfg = PipelineConfig()"""

if content.count(bad_block) == 1:
    content = content.replace(bad_block, "")
    content = content.rstrip() + "\n\n" + bad_block + "\n"
    with open(config_path, "w") as f:
        f.write(content)
    print("✓ config.py patché")
else:
    print("✓ config.py déjà bon")
```

## Cellule 4 — Mount Drive + symlink models (survie aux crashes)

```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
models_src = "/content/multimodal-alpha-signal-extractor/models"
models_dst = "/content/drive/MyDrive/alpha-signal/models"

os.makedirs(models_dst, exist_ok=True)
if os.path.exists(models_src) and not os.path.islink(models_src):
    shutil.rmtree(models_src)
if not os.path.islink(models_src):
    os.symlink(models_dst, models_src)
print("✓ models/ → Drive")
```

## Cellule 5 — WandB login

```python
import wandb
wandb.login()
```

## Cellule 6 — Génère le dataset (si pas déjà sur Drive)

```python
import os, shutil

dataset_drive = "/content/drive/MyDrive/alpha-signal/dataset"
dataset_local = "/content/multimodal-alpha-signal-extractor/dataset"

os.makedirs(dataset_drive, exist_ok=True)
if os.path.exists(dataset_local) and not os.path.islink(dataset_local):
    shutil.rmtree(dataset_local)
if not os.path.islink(dataset_local):
    os.symlink(dataset_drive, dataset_local)

if os.path.exists(f"{dataset_drive}/training_data.jsonl"):
    print("✓ Dataset chargé depuis Drive")
else:
    print("Génération du dataset...")
    !python /content/multimodal-alpha-signal-extractor/01_generate_dataset.py
```

## Cellule 7 — Lance le DPO

```python
import os
os.environ["WANDB_PROJECT"] = "alpha-signal-dpo"

!cd /content/multimodal-alpha-signal-extractor && \
  DPO_USE_UNSLOTH=0 \
  DPO_REPORT_TO=wandb \
  python 04_dpo_alignment.py
```

Les checkpoints sont sauvegardés toutes les 5 steps dans Drive (`alpha-signal/models/dpo-adapter/`).
Si la session crash, relancer la Cellule 7 : l'entraînement reprend automatiquement.

---

## Dépannage

| Erreur | Action |
|--------|--------|
| `getcwd: cannot access parent directories` | Ajouter `import os; os.chdir("/content")` avant le clone |
| `NameError: RewardModelConfig` | Exécuter la Cellule 3 (patch config.py) |
| `ModuleNotFoundError: trl` | Exécuter la Cellule 2 |
| `RecursionError` | Utiliser `DPO_USE_UNSLOTH=0` |
| `tokens: X, features: Y` | Déjà corrigé : `max_pixels=512*28*28` dans le processor |
| `divergent branches` | Utiliser `rm -rf` + clone (Cellule 1) |
