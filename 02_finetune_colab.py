#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_finetune_colab.py — Unsloth QLoRA Fine-Tuning for Google Colab (T4 GPU).

╔══════════════════════════════════════════════════════════════════╗
║  Model : Qwen2.5-VL-3B-Instruct (3 billion params)             ║
║  GPU   : Google Colab T4 (15 GB VRAM) — fits comfortably       ║
║  Time  : ~5-10 minutes for 3 epochs                            ║
║  Output: GGUF Q4_K_M (~2 GB) → import into Ollama on Mac M4    ║
╚══════════════════════════════════════════════════════════════════╝

How to use on Google Colab:
    1. Runtime → Change runtime type → T4 GPU
    2. Upload training_data.jsonl
    3. Run this script
    4. Download the .gguf file
    5. On Mac: ollama create alpha-signal-vlm -f Modelfile

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # Model — Qwen2.5-VL 3B (small, fast, fits on T4 easily)
    "base_model": "unsloth/Qwen2.5-VL-3B-Instruct",
    "base_model_hf": "Qwen/Qwen2.5-VL-3B-Instruct",  # For GGUF export
    "max_seq_length": 2048,
    "load_in_4bit": True,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 1,
    "save_steps": 50,
    "seed": 3407,

    # Paths
    "dataset_path": "training_data.jsonl",
    "lora_dir": "alpha-signal-lora",
    "merged_dir": "merged",
    "output_gguf": "alpha-signal-q4km.gguf",

    # Export
    "gguf_quantization": "Q4_K_M",
}


# ---------------------------------------------------------------------------
# 1. Dataset Loading
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> Dataset:
    """Load JSONL dataset."""
    logger.info(f"Loading dataset from {path}...")
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No samples in {path}")
    logger.info(f"  ✓ {len(samples)} training samples")
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# 2. Model Setup
# ---------------------------------------------------------------------------
def setup_model():
    """Load Qwen2.5-VL 3B and apply QLoRA."""
    logger.info(f"Loading model: {CONFIG['base_model']}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=CONFIG["base_model"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
        dtype=None,
    )

    logger.info("Applying QLoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def train(model, tokenizer, dataset: Dataset):
    """Run SFT fine-tuning with SFTConfig (required by Unsloth for VLMs)."""
    from unsloth import UnslothVisionDataCollator, is_bf16_supported

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=CONFIG["lora_dir"],
            num_train_epochs=CONFIG["num_train_epochs"],
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            learning_rate=CONFIG["learning_rate"],
            lr_scheduler_type=CONFIG["lr_scheduler_type"],
            warmup_ratio=CONFIG["warmup_ratio"],
            weight_decay=CONFIG["weight_decay"],
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=CONFIG["logging_steps"],
            save_steps=CONFIG["save_steps"],
            save_total_limit=2,
            seed=CONFIG["seed"],
            optim="adamw_8bit",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=CONFIG["max_seq_length"],
        ),
    )

    logger.info("🚀 Starting fine-tuning...")
    result = trainer.train()
    logger.info(f"✓ Loss: {result.metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Duration: {result.metrics.get('train_runtime', 0)/60:.1f} min")

    return trainer


# ---------------------------------------------------------------------------
# 4. Save LoRA adapters
# ---------------------------------------------------------------------------
def save_lora(model, tokenizer):
    """Save LoRA adapters for later merging."""
    FastVisionModel.for_inference(model)
    model.save_pretrained(CONFIG["lora_dir"])
    tokenizer.save_pretrained(CONFIG["lora_dir"])
    logger.info(f"✓ LoRA saved → {CONFIG['lora_dir']}/")


# ---------------------------------------------------------------------------
# 5. Merge LoRA + Export GGUF (shard-by-shard, memory-efficient)
# ---------------------------------------------------------------------------
def merge_and_export():
    """Merge LoRA into base model shard by shard, then convert to GGUF."""
    from safetensors.torch import load_file, save_file
    from huggingface_hub import hf_hub_download
    import subprocess

    lora_dir = CONFIG["lora_dir"]
    merged_dir = CONFIG["merged_dir"]
    base_id = CONFIG["base_model_hf"]

    # Load LoRA config
    with open(f"{lora_dir}/adapter_config.json") as f:
        acfg = json.load(f)
    scaling = acfg["lora_alpha"] / acfg["r"]
    lora_state = load_file(f"{lora_dir}/adapter_model.safetensors")

    # Build mapping
    lora_pairs = {}
    for key in lora_state:
        if ".lora_A." not in key:
            continue
        base_key = key
        for prefix in ["base_model.model.", "base_model."]:
            if base_key.startswith(prefix):
                base_key = base_key[len(prefix):]
                break
        base_key = base_key.replace(".lora_A.weight", ".weight")
        base_key = base_key.replace(".lora_A.default.weight", ".weight")
        base_key = base_key.replace("model.language_model.", "model.")
        b_key = key.replace("lora_A", "lora_B")
        if b_key in lora_state:
            lora_pairs[base_key] = (lora_state[key].float(), lora_state[b_key].float())
    logger.info(f"LoRA pairs to merge: {len(lora_pairs)}")

    # Get base model index
    idx_path = hf_hub_download(base_id, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Merge shard by shard
    os.makedirs(merged_dir, exist_ok=True)
    merged = 0
    for shard in sorted(set(weight_map.values())):
        logger.info(f"  📦 {shard}...")
        data = load_file(hf_hub_download(base_id, shard))
        for t in list(data.keys()):
            if t in lora_pairs:
                A, B = lora_pairs[t]
                data[t] = (data[t].float() + (B @ A) * scaling).half()
                merged += 1
        save_file(data, f"{merged_dir}/{shard}")
        del data
    logger.info(f"  🔗 Merged {merged}/{len(lora_pairs)} layers")

    # Copy config files
    for fname in ["config.json", "generation_config.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt", "tokenizer.json", "chat_template.json",
                  "preprocessor_config.json", "model.safetensors.index.json"]:
        try:
            shutil.copy(hf_hub_download(base_id, fname), f"{merged_dir}/{fname}")
        except Exception:
            pass

    # Clean quantization_config
    cfg_path = f"{merged_dir}/config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    logger.info("✓ Merged model ready")

    # Convert to GGUF
    logger.info("Converting to GGUF...")
    subprocess.run(["pip", "install", "-q", "gguf", "sentencepiece", "protobuf"], check=True)

    if not os.path.exists("llama.cpp"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp.git"], check=True)

    # HF → F16
    subprocess.run([
        "python", "llama.cpp/convert_hf_to_gguf.py", merged_dir,
        "--outfile", "model-f16.gguf", "--outtype", "f16"
    ], check=True)

    # Clean merged to free disk
    shutil.rmtree(merged_dir)

    # Build quantizer
    subprocess.run("cd llama.cpp && cmake -B build && cmake --build build "
                   "--target llama-quantize -j$(nproc)",
                   shell=True, check=True, capture_output=True)

    # F16 → Q4_K_M
    quant = CONFIG["gguf_quantization"]
    output = CONFIG["output_gguf"]
    subprocess.run([
        "llama.cpp/build/bin/llama-quantize",
        "model-f16.gguf", output, quant
    ], check=True)

    os.remove("model-f16.gguf")

    size_mb = os.path.getsize(output) / 1024**2
    logger.info(f"🎉 GGUF ready: {output} ({size_mb:.0f} MB)")
    logger.info("")
    logger.info("=" * 60)
    logger.info("  NEXT STEPS (on your Mac M4):")
    logger.info("  1. Download alpha-signal-q4km.gguf")
    logger.info("  2. Create Modelfile:")
    logger.info("     echo 'FROM ./alpha-signal-q4km.gguf' > Modelfile")
    logger.info("  3. Import: ollama create alpha-signal-vlm -f Modelfile")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Full pipeline: load → train → save → merge → export GGUF."""
    logger.info("=" * 60)
    logger.info("  Multimodal Alpha-Signal Extractor")
    logger.info("  Fine-Tuning Qwen2.5-VL 3B (Google Colab T4)")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available! Use Google Colab with T4 GPU.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info(f"GPU: {gpu} ({vram:.1f} GB)")

    # 1. Dataset
    dataset = load_dataset(CONFIG["dataset_path"])

    # 2. Model
    model, tokenizer = setup_model()

    # 3. Train
    trainer = train(model, tokenizer, dataset)

    # 4. Save LoRA
    save_lora(model, tokenizer)

    # 5. Free VRAM
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Merge + GGUF
    merge_and_export()

    logger.info("✓ Pipeline complete!")


if __name__ == "__main__":
    main()
