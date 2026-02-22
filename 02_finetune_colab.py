#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_finetune_colab.py — Unsloth QLoRA Fine-Tuning Script for Google Colab.

╔══════════════════════════════════════════════════════════════════╗
║  This script is designed to run on Google Colab (Free T4 GPU).  ║
║  It will NOT work on Apple Silicon / M-series Macs.             ║
║                                                                 ║
║  Usage on Colab:                                                ║
║  1. Upload training_data.jsonl to Colab                         ║
║  2. pip install unsloth                                         ║
║  3. Run this script                                             ║
║  4. Download the GGUF model                                     ║
║  5. Import into Ollama on your M4 Mac                           ║
╚══════════════════════════════════════════════════════════════════╝

Steps after fine-tuning on Colab:
    1. Download the .gguf file from Colab
    2. On your Mac M4, create an Ollama Modelfile:
       echo 'FROM ./model-q4_k_m.gguf' > Modelfile
    3. Import: ollama create alpha-signal-vlm -f Modelfile
    4. Update config.py: ollama_vlm_model = "alpha-signal-vlm"

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# ============================================================================
# 0. COLAB SETUP — Run these cells first in Colab
# ============================================================================
# Cell 1: Install Unsloth
# !pip install unsloth
# !pip install --no-deps trl peft accelerate bitsandbytes

# Cell 2: Upload your dataset
# from google.colab import files
# uploaded = files.upload()  # Upload training_data.jsonl

# ============================================================================

import torch
from datasets import Dataset
from trl import SFTTrainer
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
# Configuration (inline for Colab — no external config.py needed)
# ---------------------------------------------------------------------------
CONFIG = {
    # Model
    "base_model": "unsloth/Qwen2-VL-7B-Instruct",
    "max_seq_length": 2048,
    "load_in_4bit": True,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 5,
    "save_steps": 50,
    "seed": 42,

    # Paths
    "dataset_path": "training_data.jsonl",
    "output_dir": "alpha-signal-vlm",

    # Export
    "save_gguf": True,
    "gguf_quantization": "q4_k_m",
}


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(jsonl_path: str) -> Dataset:
    """Load the Unsloth-compatible JSONL file into a HuggingFace Dataset.

    Args:
        jsonl_path: Path to the JSONL training file.

    Returns:
        A HuggingFace Dataset object.
    """
    logger.info(f"Loading dataset from {jsonl_path}...")
    samples: list[dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: {e}")

    if not samples:
        raise ValueError(f"No valid samples in {jsonl_path}")

    logger.info(f"✓ Loaded {len(samples)} training samples")
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def setup_model():
    """Load the base VLM and apply QLoRA adapters via Unsloth.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model: {CONFIG['base_model']}")

    # 1. Load pre-trained VLM in 4-bit
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=CONFIG["base_model"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
        dtype=None,  # Auto-detect
    )

    # 2. Apply QLoRA adapters
    logger.info("Applying QLoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
        use_rslora=False,
        loftq_config=None,
        # Fine-tune vision encoder projection layers
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, tokenizer, dataset: Dataset):
    """Run supervised fine-tuning.

    Args:
        model: PEFT-wrapped VLM.
        tokenizer: Tokenizer/processor.
        dataset: Training dataset.
    """
    from unsloth import UnslothVisionDataCollator, is_bf16_supported
    from trl import SFTConfig

    output_dir = CONFIG["output_dir"]

    # ⚠️ CRITICAL: Use SFTConfig (NOT TrainingArguments)
    # dataset_text_field, dataset_kwargs, max_seq_length go INSIDE SFTConfig
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=CONFIG["num_train_epochs"],
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            learning_rate=CONFIG["learning_rate"],
            lr_scheduler_type=CONFIG["lr_scheduler_type"],
            warmup_ratio=CONFIG["warmup_ratio"],
            weight_decay=CONFIG["weight_decay"],
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            gradient_checkpointing=CONFIG["gradient_checkpointing"],
            logging_steps=CONFIG["logging_steps"],
            save_steps=CONFIG["save_steps"],
            save_total_limit=3,
            seed=CONFIG["seed"],
            dataloader_num_workers=2,
            report_to="none",
            optim="adamw_8bit",
            remove_unused_columns=False,
            # === These 3 lines MUST be in SFTConfig, not SFTTrainer ===
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=CONFIG["max_seq_length"],
        ),
    )

    logger.info("=" * 60)
    logger.info("  Starting fine-tuning on Google Colab T4...")
    logger.info(f"  Epochs: {CONFIG['num_train_epochs']}")
    logger.info(f"  Batch:  {CONFIG['per_device_train_batch_size']}")
    logger.info(f"  LR:     {CONFIG['learning_rate']}")
    logger.info("=" * 60)

    result = trainer.train()
    logger.info(f"Training complete. Metrics: {result.metrics}")

    # Save checkpoint
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(model, tokenizer):
    """Export the fine-tuned model to safetensors + GGUF.

    The GGUF file is what you'll download and import into Ollama on your Mac.

    Args:
        model: Fine-tuned PEFT model.
        tokenizer: Tokenizer.
    """
    output_dir = CONFIG["output_dir"]

    # Merged 16-bit (safetensors)
    merged_dir = f"{output_dir}_merged_16bit"
    logger.info(f"Saving merged 16-bit model → {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # GGUF for Ollama
    if CONFIG["save_gguf"]:
        gguf_dir = f"{output_dir}_gguf"
        quant = CONFIG["gguf_quantization"]
        logger.info(f"Exporting GGUF ({quant}) → {gguf_dir}")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
        logger.info(f"✓ GGUF ready at {gguf_dir}/")
        logger.info("")
        logger.info("=" * 60)
        logger.info("  NEXT STEPS (on your Mac M4):")
        logger.info("  1. Download the .gguf file from Colab")
        logger.info("  2. Create Modelfile:")
        logger.info("     echo 'FROM ./model-q4_k_m.gguf' > Modelfile")
        logger.info("  3. Import into Ollama:")
        logger.info("     ollama create alpha-signal-vlm -f Modelfile")
        logger.info("  4. Update config.py:")
        logger.info('     ollama_vlm_model = "alpha-signal-vlm"')
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full fine-tuning pipeline on Colab."""
    logger.info("=" * 60)
    logger.info("  Multimodal Alpha-Signal Extractor")
    logger.info("  Fine-Tuning with Unsloth (Google Colab T4)")
    logger.info("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error(
            "CUDA not available! This script must run on Google Colab "
            "with a T4 GPU. Go to Runtime → Change runtime type → T4 GPU."
        )
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # 1. Load dataset
    dataset = load_jsonl_dataset(CONFIG["dataset_path"])

    # 2. Setup model + LoRA
    model, tokenizer = setup_model()

    # 3. Train
    FastVisionModel.for_training(model)
    train(model, tokenizer, dataset)

    # 4. Export
    FastVisionModel.for_inference(model)
    export_model(model, tokenizer)

    logger.info("✓ Fine-tuning complete!")


if __name__ == "__main__":
    main()
