"""
config.py — Centralized Configuration for the Multimodal Alpha-Signal Extractor.

All hyperparameters, paths, and API endpoints are defined here as frozen
dataclasses to ensure immutability and type-safety across the pipeline.

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
CHARTS_DIR = DATASET_DIR / "charts"


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for synthetic dataset generation (Step 1)."""

    ticker: str = "AAPL"
    period: str = "2y"
    interval: str = "1d"
    window_size: int = 60          # Trading days per chart
    stride: int = 5                # Sliding window step
    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    forward_return_days: int = 5   # Days to look ahead for label
    output_jsonl: Path = DATASET_DIR / "training_data.jsonl"
    chart_dpi: int = 100
    chart_style: str = "charles"   # mplfinance style


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for Unsloth QLoRA fine-tuning (Step 2)."""

    base_model: str = "unsloth/Qwen2-VL-7B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Training arguments
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_steps: int = 50
    seed: int = 42
    output_dir: Path = MODELS_DIR / "qwen2-vl-alpha-signal"
    # Export
    save_gguf: bool = True
    gguf_quantization: str = "q4_k_m"


@dataclass(frozen=True)
class VLLMConfig:
    """Configuration for the vLLM inference server (Step 3)."""

    model_path: str = str(MODELS_DIR / "qwen2-vl-alpha-signal")
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    dtype: str = "auto"
    quantization: str | None = None  # "awq", "gptq", or None
    trust_remote_code: bool = True
    api_key: str = "alpha-signal-key"  # Simple auth for local use
    max_num_seqs: int = 64            # Max concurrent sequences
    enable_prefix_caching: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the LangChain orchestrator (Step 4).

    On Apple Silicon (M4), both VLM and text models are served via Ollama.
    On CUDA machines, the VLM can be served via vLLM instead.
    """

    # Ollama endpoint for VLM (vision + text)
    vlm_provider: str = "ollama"  # "ollama" or "vllm"
    # Default: generic model. After fine-tuning, change to "alpha-signal-vlm"
    # Note: Ollama only supports llama3.2-vision for VLM inference currently.
    # Qwen2.5-VL GGUFs crash due to unsupported multi-dim positional embeddings.
    ollama_vlm_model: str = "llama3.2-vision:11b"

    # vLLM fallback (for CUDA machines only)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "alpha-signal-key"
    vllm_model_name: str = "qwen2-vl-alpha-signal"

    # Shared VLM settings
    vlm_temperature: float = 0.1
    vlm_max_tokens: int = 1024

    # Ollama endpoint (text-only sentiment LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    ollama_temperature: float = 0.0

    # Retry policy
    max_retries: int = 3
    retry_wait_seconds: float = 2.0

    # Logging
    log_level: str = "INFO"


# ============================================================================
# Singleton instances (import directly)
# ============================================================================
dataset_cfg = DatasetConfig()
training_cfg = TrainingConfig()
vllm_cfg = VLLMConfig()
pipeline_cfg = PipelineConfig()
