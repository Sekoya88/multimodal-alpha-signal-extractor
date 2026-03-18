"""di_container.py — Dependency Injection Container.

Wires up all ports and adapters to instantiate the Application use cases.
Reads from configuration to pick the correct implementation.
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on path when running as installed script
# di_container.py -> presentation/ -> alpha_signal/ -> src/ -> project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import pipeline_cfg, reward_model_cfg

from alpha_signal.application.usecases import AnalyzeMarketUseCase
from alpha_signal.infrastructure.adapters.llm_adapters import (
    LlamaCppVlmAdapter,
    OllamaSentimentAdapter,
    OllamaVlmAdapter,
    VllmVlmAdapter,
)
from alpha_signal.infrastructure.adapters.mplfinance_adapter import MplfinanceAdapter
from alpha_signal.infrastructure.adapters.yfinance_adapter import YFinanceAdapter


def _build_reward_scorer() -> Optional["RewardScorerPort"]:
    """Try to build the reward scorer if weights are available."""
    from alpha_signal.application.ports import RewardScorerPort

    weights_path = reward_model_cfg.output_dir / "mlp_head.pt"
    if not weights_path.exists():
        return None

    try:
        from alpha_signal.infrastructure.adapters.reward_scorer_adapter import RewardScorerAdapter
        adapter = RewardScorerAdapter(
            hidden_dim=reward_model_cfg.hidden_dim,
            dropout=reward_model_cfg.dropout,
        )
        adapter.load_weights(weights_path)
        return adapter
    except Exception:
        return None


def build_analyze_market_usecase(output_dir: Path) -> AnalyzeMarketUseCase:
    """Builds and wires the AnalyzeMarketUseCase with the correct adapters based on config.
    
    Args:
        output_dir: Where to save the generated charts.
        
    Returns:
        A ready-to-use AnalyzeMarketUseCase instance.
    """
    # 1. Instantiate Data/Rendering Adapters
    market_data_port = YFinanceAdapter()
    news_port = market_data_port  # YFinanceAdapter implements both
    chart_renderer_port = MplfinanceAdapter()

    # 2. Instantiate Sentiment Adapter
    sentiment_port = OllamaSentimentAdapter(
        base_url=pipeline_cfg.ollama_base_url,
        model_name=pipeline_cfg.ollama_model,
        temperature=pipeline_cfg.ollama_temperature,
    )

    # 3. Instantiate VLM Adapter based on configuration
    if pipeline_cfg.vlm_provider == "ollama":
        vlm_port = OllamaVlmAdapter(
            base_url=pipeline_cfg.ollama_base_url,
            model_name=pipeline_cfg.ollama_vlm_model,
            temperature=pipeline_cfg.vlm_temperature,
        )
    elif pipeline_cfg.vlm_provider == "vllm":
        vlm_port = VllmVlmAdapter(
            base_url=pipeline_cfg.vllm_base_url,
            api_key=pipeline_cfg.vllm_api_key,
            model_name=pipeline_cfg.vllm_model_name,
            temperature=pipeline_cfg.vlm_temperature,
            max_tokens=pipeline_cfg.vlm_max_tokens,
        )
    elif pipeline_cfg.vlm_provider == "llama_cpp":
        vlm_port = LlamaCppVlmAdapter(
            model_path=pipeline_cfg.llama_cpp_model_path,
            mmproj_path=pipeline_cfg.llama_cpp_mmproj_path,
            n_gpu_layers=pipeline_cfg.llama_cpp_n_gpu_layers,
            n_ctx=pipeline_cfg.llama_cpp_n_ctx,
            temperature=pipeline_cfg.vlm_temperature,
            max_tokens=pipeline_cfg.vlm_max_tokens,
        )
    else:
        raise ValueError(f"Unknown VLM provider: {pipeline_cfg.vlm_provider}")

    # 4. Optionally load reward scorer (6th pipeline node)
    reward_scorer_port = _build_reward_scorer()

    # 5. Inject dependencies into Use Case
    use_case = AnalyzeMarketUseCase(
        market_data_port=market_data_port,
        news_port=news_port,
        chart_renderer_port=chart_renderer_port,
        vlm_port=vlm_port,
        sentiment_port=sentiment_port,
        output_dir=output_dir,
        reward_scorer_port=reward_scorer_port,
    )
    
    return use_case
