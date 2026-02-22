#!/usr/bin/env python3
"""
02_finetune_vlm.py — QLoRA Fine-Tuning of a Vision-Language Model with Unsloth.

Loads a pre-trained VLM (Qwen2-VL-7B-Instruct), applies QLoRA adapters on
attention and vision projection layers, trains on the synthetic multimodal
dataset generated in Step 1, and exports the model in safetensors + GGUF.

Requirements:
    - NVIDIA GPU with ≥16 GB VRAM (A100/H100 recommended)
    - CUDA toolkit installed
    - pip install unsloth

Usage:
    python 02_finetune_vlm.py

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastVisionModel

from config import DATASET_DIR, training_cfg

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
# Dataset Loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(jsonl_path: Path) -> Dataset:
    """Load the Unsloth-compatible JSONL file into a HuggingFace Dataset.

    Each line in the JSONL file is a conversation with `messages` containing
    multimodal content blocks (image + text).

    Args:
        jsonl_path: Path to the JSONL training file.

    Returns:
        A HuggingFace `Dataset` object.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If the file is empty or malformed.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    logger.info(f"Loading dataset from {jsonl_path}...")
    samples: list[dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")

    if not samples:
        raise ValueError(f"No valid samples found in {jsonl_path}")

    logger.info(f"Loaded {len(samples)} training samples")
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer() -> tuple[Any, Any]:
    """Load the base VLM and apply QLoRA adapters via Unsloth.

    This function:
      1. Loads the pre-trained VLM (Qwen2-VL) in 4-bit quantization.
      2. Applies LoRA adapters to attention layers (q, k, v, o projections)
         and feed-forward layers (gate, up, down projections).
      3. Returns the PEFT model and its tokenizer/processor.

    Returns:
        Tuple of (model, tokenizer).

    Notes:
        - Unsloth's `FastVisionModel` handles quantization, device mapping,
          and PEFT injection automatically.
        - The `finetune_vision_layers` flag also applies adapters to the
          vision encoder's projection layers for cross-modal alignment.
    """
    logger.info(f"Loading base model: {training_cfg.base_model}")
    logger.info(f"  4-bit quantization : {training_cfg.load_in_4bit}")
    logger.info(f"  Max seq length     : {training_cfg.max_seq_length}")

    # ---- 1. Load pre-trained VLM ----
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=training_cfg.base_model,
        max_seq_length=training_cfg.max_seq_length,
        load_in_4bit=training_cfg.load_in_4bit,
        dtype=None,  # Auto-detect (bf16 on Ampere+, fp16 otherwise)
    )

    # ---- 2. Apply QLoRA adapters ----
    logger.info("Applying QLoRA adapters...")
    logger.info(f"  LoRA r       : {training_cfg.lora_r}")
    logger.info(f"  LoRA alpha   : {training_cfg.lora_alpha}")
    logger.info(f"  Target modules: {training_cfg.target_modules}")

    model = FastVisionModel.get_peft_model(
        model,
        r=training_cfg.lora_r,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        target_modules=list(training_cfg.target_modules),
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=training_cfg.seed,
        use_rslora=False,
        loftq_config=None,
        # Fine-tune vision encoder projection layers too
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"  Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Data Collation (Multimodal)
# ---------------------------------------------------------------------------

def create_data_collator(tokenizer: Any) -> Any:
    """Create a data collator that handles multimodal conversation formatting.

    Uses Unsloth's built-in `UnslothVisionDataCollator` which handles:
      - Chat template application
      - Image preprocessing and resizing
      - Padding and batching of mixed image+text inputs

    Args:
        tokenizer: The model's tokenizer/processor.

    Returns:
        A callable data collator.
    """
    from unsloth import UnslothVisionDataCollator

    return UnslothVisionDataCollator(model=None, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
) -> None:
    """Run the SFTTrainer fine-tuning loop.

    Args:
        model: The PEFT-wrapped VLM.
        tokenizer: Corresponding tokenizer/processor.
        dataset: HuggingFace Dataset with conversation data.
    """
    output_dir = str(training_cfg.output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.num_train_epochs,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        warmup_ratio=training_cfg.warmup_ratio,
        weight_decay=training_cfg.weight_decay,
        bf16=training_cfg.bf16,
        fp16=training_cfg.fp16,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=3,
        seed=training_cfg.seed,
        dataloader_num_workers=4,
        report_to="none",  # Disable W&B / MLflow unless configured
        optim="adamw_8bit",
        remove_unused_columns=False,  # Required for multimodal
    )

    # SFTTrainer from TRL — handles conversation-style fine-tuning
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=create_data_collator(tokenizer),
        dataset_text_field=None,   # Using messages format, not raw text
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=training_cfg.max_seq_length,
    )

    logger.info("=" * 70)
    logger.info("  Starting fine-tuning...")
    logger.info(f"  Epochs               : {training_cfg.num_train_epochs}")
    logger.info(f"  Batch size (per GPU) : {training_cfg.per_device_train_batch_size}")
    logger.info(f"  Gradient accum steps : {training_cfg.gradient_accumulation_steps}")
    logger.info(f"  Learning rate        : {training_cfg.learning_rate}")
    logger.info(f"  Scheduler            : {training_cfg.lr_scheduler_type}")
    logger.info(f"  Output dir           : {output_dir}")
    logger.info("=" * 70)

    # ---- Train ----
    train_result = trainer.train()

    # Log metrics
    metrics = train_result.metrics
    logger.info(f"Training complete. Metrics: {metrics}")

    # ---- Save final checkpoint ----
    logger.info("Saving final model checkpoint...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(model: Any, tokenizer: Any) -> None:
    """Export the fine-tuned model to safetensors and optionally GGUF.

    Args:
        model: The fine-tuned PEFT model.
        tokenizer: Corresponding tokenizer/processor.
    """
    output_dir = str(training_cfg.output_dir)

    # ---- Save merged 16-bit model (safetensors) ----
    safetensors_dir = f"{output_dir}_16bit"
    logger.info(f"Saving merged 16-bit model to {safetensors_dir}...")
    model.save_pretrained_merged(
        safetensors_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    # ---- Optionally export to GGUF ----
    if training_cfg.save_gguf:
        gguf_dir = f"{output_dir}_gguf"
        quant = training_cfg.gguf_quantization
        logger.info(f"Exporting GGUF ({quant}) to {gguf_dir}...")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method=quant,
        )
        logger.info(f"GGUF export complete: {gguf_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full fine-tuning pipeline."""
    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal Extractor — VLM Fine-Tuning (Unsloth)")
    logger.info("=" * 70)

    # Verify CUDA availability
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. This script requires an NVIDIA GPU with "
            "CUDA toolkit installed. Exiting."
        )
        sys.exit(1)

    logger.info(f"CUDA device : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM        : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---- 1. Load dataset ----
    jsonl_path = DATASET_DIR / "training_data.jsonl"
    dataset = load_jsonl_dataset(jsonl_path)

    # ---- 2. Setup model + LoRA ----
    model, tokenizer = setup_model_and_tokenizer()

    # ---- 3. Train ----
    FastVisionModel.for_training(model)  # Enable training mode
    train(model, tokenizer, dataset)

    # ---- 4. Export ----
    FastVisionModel.for_inference(model)  # Switch to inference mode for export
    export_model(model, tokenizer)

    logger.info("=" * 70)
    logger.info("  Fine-tuning pipeline complete ✓")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
