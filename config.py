"""config.py — Centralized configuration for the Multimodal Alpha-Signal Extractor.

All configurable parameters live here so that the pipeline scripts
remain clean and DRY.
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
class DPOConfig:
    """Configuration for DPO alignment (Sprint 1)."""

    base_model: str = "unsloth/Qwen2.5-VL-3B-Instruct"
    sft_adapter_path: Path | None = None  # Optional: load SFT LoRA before DPO
    max_seq_length: int = 1024       # Reduced for Colab T4 OOM
    load_in_4bit: bool = True
    # LoRA (same as SFT)
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # DPO
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    beta: float = 0.1  # DPO beta
    max_prompt_length: int = 512     # Reduced for Colab
    max_length: int = 1024           # Reduced for Colab
    seed: int = 42
    output_dir: Path = MODELS_DIR / "dpo-adapter"
    # Paths
    dataset_path: Path = DATASET_DIR / "training_data.jsonl"
    dpo_pairs_path: Path = DATASET_DIR / "dpo_pairs.jsonl"


@dataclass(frozen=True)
class RewardModelConfig:
    """Configuration for the Visual Reward Model (Sprint 2)."""

    base_model: str = "unsloth/Qwen2.5-VL-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    # MLP head
    hidden_dim: int = 256          # MLP hidden layer dimension
    dropout: float = 0.1
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    # Reward labeling
    reward_threshold: float = 0.02  # 2% return → reward 1.0
    # Paths
    output_dir: Path = MODELS_DIR / "reward-model"
    training_data_path: Path = DATASET_DIR / "reward_training_data.jsonl"


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training (Sprint 3)."""

    base_model: str = "unsloth/Qwen2.5-VL-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # GRPO specifics
    group_size: int = 8                  # N predictions per chart
    temperature: float = 0.7            # Temperature for diverse sampling
    epsilon: float = 0.2                # PPO clipping epsilon
    reward_weight_direction: float = 0.6
    reward_weight_calibration: float = 0.4
    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    seed: int = 42
    # Logging
    log_to_wandb: bool = False
    log_csv_path: Path = DATASET_DIR / "grpo_reward_curves.csv"
    # Paths
    output_dir: Path = MODELS_DIR / "grpo-adapter"
    dataset_path: Path = DATASET_DIR / "training_data.jsonl"


@dataclass(frozen=True)
class TemporalConfig:
    """Configuration for Temporal Multi-Frame Extension (Sprint 4)."""

    n_frames: int = 8                   # Number of consecutive windows
    window_size: int = 60               # Trading days per chart
    stride: int = 5                     # Days between frames
    # Position embeddings
    position_embedding_dim: int = 64    # Learned temporal position embedding dim
    # Model
    base_model: str = "unsloth/Qwen2.5-VL-3B-Instruct"
    max_seq_length: int = 4096          # Larger context for multi-image
    load_in_4bit: bool = True
    # Benchmark
    benchmark_holdout_ratio: float = 0.2
    seed: int = 42
    # Paths
    output_dir: Path = MODELS_DIR / "temporal-extension"
    benchmark_results_path: Path = DATASET_DIR / "temporal_benchmark.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for Production Inference Benchmark (Sprint 5)."""

    # Redis Cache
    redis_url: str = "redis://localhost:6379/1"
    cache_ttl_seconds: int = 3600  # 1 hour
    # vLLM Optimization Defaults
    tensor_parallel_size: int = 1
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    # Speculative Decoding
    speculative_draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    speculative_draft_tensor_parallel_size: int = 1
    num_speculative_tokens: int = 4  # gamma
    # Benchmark execution
    concurrent_users: list[int] = field(default_factory=lambda: [1, 4, 16, 32])
    requests_per_user: int = 5
    seed: int = 42
    # Paths
    benchmark_results_path: Path = DATASET_DIR / "inference_benchmark.json"
    benchmark_plot_path: Path = DATASET_DIR / "inference_benchmark_plot.png"


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

    # VLM backend: "ollama", "vllm", or "llama_cpp"
    # - ollama: llama3.2-vision (only VLM Ollama supports)
    # - llama_cpp: direct GGUF loading (for fine-tuned Qwen2.5-VL)
    # - vllm: CUDA machines only
    vlm_provider: str = "llama_cpp"  # ← uses fine-tuned GGUF
    ollama_vlm_model: str = "llama3.2-vision:11b"

    # LLAMA.CPP BACKEND (Apple Silicon M4 - Fine-Tuned GGUF)
    # Override via LLAMA_CPP_MODEL_PATH / LLAMA_CPP_MMPROJ_PATH (used by HF Spaces entrypoint).
    # --------------------------------------------------------------------------
    llama_cpp_model_path: str = os.environ.get(
        "LLAMA_CPP_MODEL_PATH",
        os.path.expanduser("~/Downloads/alpha-signal-q4km.gguf"),
    )
    llama_cpp_mmproj_path: str = os.environ.get(
        "LLAMA_CPP_MMPROJ_PATH",
        str(PROJECT_ROOT / "models" / "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"),
    )
    llama_cpp_n_gpu_layers: int = -1  # -1 for all layers on Metal
    llama_cpp_n_ctx: int = 8192       # Vision tokens require a large context window
    # vLLM fallback (for CUDA machines only)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "alpha-signal-key"
    vllm_model_name: str = "qwen2-vl-alpha-signal"

    # Shared VLM settings
    vlm_temperature: float = 0.1
    vlm_max_tokens: int = 1024

    # Ollama endpoint (text-only sentiment LLM)
    # Override via OLLAMA_MODEL (e.g. qwen2.5:0.5b for HF free-tier).
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3:8b")
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
dpo_cfg = DPOConfig()
reward_model_cfg = RewardModelConfig()
grpo_cfg = GRPOConfig()
temporal_cfg = TemporalConfig()
benchmark_cfg = BenchmarkConfig()
vllm_cfg = VLLMConfig()
pipeline_cfg = PipelineConfig()
