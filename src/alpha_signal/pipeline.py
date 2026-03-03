"""pipeline.py — LangChain orchestrator (refactored from 04_langchain_pipeline.py).

This module provides the core pipeline logic that can be imported by both
the CLI scripts and the Streamlit app. It delegates to shared schemas and
config modules.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .schemas import (
    SentimentResult,
    TradeAction,
    TradingDecision,
    TradingSignal,
)

logger = logging.getLogger(__name__)

# Module-level model cache for llama.cpp
_llama_model: Any = None


# ============================================================================
# Image Utilities
# ============================================================================

def encode_image_to_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_as_data_uri(image_path: Path) -> str:
    """Read an image and return a data URI for OpenAI-compatible APIs."""
    path = Path(image_path).resolve()
    suffix = path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(suffix, "image/png")
    b64 = encode_image_to_base64(path)
    return f"data:{mime_type};base64,{b64}"


# ============================================================================
# VLM Chain
# ============================================================================

def _build_vlm_system_prompt(format_instructions: str) -> str:
    return (
        "Tu es un analyste quantitatif senior dans un hedge fund de premier plan "
        "(type Jane Street). Tu analyses des graphiques financiers en chandeliers "
        "japonais (avec RSI et bandes de Bollinger) et le contexte d'actualité "
        "pour générer des signaux de trading.\n\n"
        "RÈGLES STRICTES:\n"
        "1. Réponds UNIQUEMENT avec un objet JSON valide.\n"
        "2. Pas de texte avant ou après le JSON.\n"
        "3. Utilise les indicateurs techniques visibles sur le graphique.\n"
        "4. Le stop_loss et take_profit doivent être réalistes.\n\n"
        f"{format_instructions}"
    )


def _parse_json_response(
    raw_content: str,
    parser: PydanticOutputParser,
) -> TradingSignal:
    """Parse JSON from model response, handling common formatting issues."""
    content = raw_content.strip()

    for attempt_fn in [
        lambda: parser.parse(content),
        lambda: parser.parse(content.split("```json")[1].split("```")[0].strip())
        if "```json" in content else None,
        lambda: parser.parse(content.split("```")[1].split("```")[0].strip())
        if "```" in content else None,
        lambda: parser.parse(content[content.find("{"):content.rfind("}") + 1])
        if "{" in content else None,
    ]:
        try:
            result = attempt_fn()
            if result is not None:
                return result
        except Exception:
            continue

    raise ValueError(
        f"Could not parse VLM response as TradingSignal JSON. "
        f"Raw response (first 500 chars): {content[:500]}"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        f"VLM call failed, retrying ({retry_state.attempt_number}/3)..."
    ),
)
async def run_vlm_chain(
    image_path: Path,
    news_text: str,
    vlm_provider: str = "llama_cpp",
    config: Any = None,
    on_status: Any = None,
) -> TradingSignal:
    """Execute the VLM chain with a chart image and news context.

    Args:
        image_path: Path to the candlestick chart image.
        news_text: Financial news context string.
        vlm_provider: Backend to use ('ollama', 'llama_cpp', or 'vllm').
        config: PipelineConfig object (uses default if None).
        on_status: Optional callback(status_str) for progress updates.

    Returns:
        Parsed TradingSignal object.
    """
    if config is None:
        from config import pipeline_cfg
        config = pipeline_cfg

    parser = PydanticOutputParser(pydantic_object=TradingSignal)
    format_instructions = parser.get_format_instructions()
    system_prompt = _build_vlm_system_prompt(format_instructions)

    user_text = (
        f"Analyse le graphique financier ci-dessus et le contexte "
        f"d'actualité suivant :\n\n"
        f"**Actualité** : {news_text}\n\n"
        f"Génère un signal de trading structuré au format JSON demandé."
    )

    if on_status:
        on_status("🔍 VLM analysis in progress...")

    if vlm_provider == "ollama":
        logger.info(f"Invoking VLM via Ollama ({config.ollama_vlm_model})...")
        llm = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.ollama_vlm_model,
            temperature=config.vlm_temperature,
        )
        img_data_uri = encode_image_as_data_uri(image_path)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": img_data_uri}},
                {"type": "text", "text": user_text},
            ]),
        ]
        response = await llm.ainvoke(messages)
        raw_content = response.content

    elif vlm_provider == "llama_cpp":
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        gguf_path = config.llama_cpp_model_path
        mmproj_path = config.llama_cpp_mmproj_path
        logger.info(f"Invoking VLM via llama.cpp ({Path(gguf_path).name})...")

        global _llama_model
        if _llama_model is None:
            logger.info("  Loading GGUF model + mmproj (first call, ~10s)...")
            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
            _llama_model = Llama(
                model_path=gguf_path,
                chat_handler=chat_handler,
                n_gpu_layers=config.llama_cpp_n_gpu_layers,
                n_ctx=config.llama_cpp_n_ctx,
                verbose=False,
            )
            logger.info("  ✓ Model loaded")

        img_data_uri = encode_image_as_data_uri(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_data_uri}},
                {"type": "text", "text": user_text},
            ]}
        ]
        result = _llama_model.create_chat_completion(
            messages=messages,
            max_tokens=config.vlm_max_tokens,
            temperature=config.vlm_temperature,
        )
        raw_content = result["choices"][0]["message"]["content"]

    else:
        from langchain_openai import ChatOpenAI

        logger.info(f"Invoking VLM via vLLM ({config.vllm_model_name})...")
        llm = ChatOpenAI(
            base_url=config.vllm_base_url,
            api_key=config.vllm_api_key,
            model=config.vllm_model_name,
            temperature=config.vlm_temperature,
            max_tokens=config.vlm_max_tokens,
        )
        img_data_uri = encode_image_as_data_uri(image_path)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": img_data_uri, "detail": "high"}},
                {"type": "text", "text": user_text},
            ]),
        ]
        response = await llm.ainvoke(messages)
        raw_content = response.content

    logger.debug(f"Raw VLM response: {raw_content[:300]}...")
    signal = _parse_json_response(raw_content, parser)
    logger.info(f"VLM Signal: {signal.action.value} (confidence: {signal.confidence:.2f})")

    if on_status:
        on_status(f"✅ VLM: {signal.action.value} ({signal.confidence:.0%})")

    return signal


# ============================================================================
# Sentiment Chain
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        f"Sentiment call failed, retrying ({retry_state.attempt_number}/3)..."
    ),
)
async def run_sentiment_chain(
    news_text: str,
    config: Any = None,
    on_status: Any = None,
) -> SentimentResult:
    """Execute the sentiment chain on financial news via Ollama.

    Args:
        news_text: Financial news text to analyze.
        config: PipelineConfig object (uses default if None).
        on_status: Optional callback(status_str) for progress updates.

    Returns:
        Parsed SentimentResult object.
    """
    if config is None:
        from config import pipeline_cfg
        config = pipeline_cfg

    parser = PydanticOutputParser(pydantic_object=SentimentResult)
    format_instructions = parser.get_format_instructions()

    llm = ChatOllama(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        temperature=config.ollama_temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Tu es un analyste de sentiment financier expert. "
            "Analyse le texte d'actualité et extrais le sentiment du marché. "
            "Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte "
            "avant ou après.\n\n"
            "{format_instructions}"
        ),
        (
            "human",
            "Analyse le sentiment de l'actualité financière suivante :\n\n"
            "{news_text}\n\n"
            "Réponds en JSON strict."
        ),
    ])

    chain = prompt | llm

    if on_status:
        on_status("🧠 Sentiment analysis in progress...")

    logger.info(f"Invoking Sentiment chain (Ollama {config.ollama_model})...")
    response = await chain.ainvoke({
        "format_instructions": format_instructions,
        "news_text": news_text,
    })

    content = response.content.strip()
    try:
        sentiment = parser.parse(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            sentiment = parser.parse(content[start:end])
        else:
            raise

    logger.info(f"Sentiment: {sentiment.sentiment} (intensity: {sentiment.intensity:.2f})")

    if on_status:
        on_status(f"✅ Sentiment: {sentiment.sentiment} ({sentiment.intensity:.0%})")

    return sentiment


# ============================================================================
# Signal Merger
# ============================================================================

def merge_signals(
    vlm_signal: TradingSignal,
    sentiment: SentimentResult,
    vlm_provider: str = "llama_cpp",
    config: Any = None,
) -> TradingDecision:
    """Merge VLM trading signal with text sentiment into a final decision.

    Args:
        vlm_signal: Output from the VLM chain.
        sentiment: Output from the sentiment chain.
        vlm_provider: Which VLM backend was used.
        config: PipelineConfig for meta info.

    Returns:
        Unified TradingDecision.
    """
    if config is None:
        from config import pipeline_cfg
        config = pipeline_cfg

    sentiment_direction = {
        "BULLISH": TradeAction.BUY,
        "BEARISH": TradeAction.SELL,
        "NEUTRAL": TradeAction.HOLD,
    }
    sentiment_bias = sentiment_direction.get(sentiment.sentiment, TradeAction.HOLD)

    if vlm_signal.action == TradeAction.HOLD:
        final_action = TradeAction.HOLD
        final_confidence = vlm_signal.confidence * 0.8
    elif vlm_signal.action == sentiment_bias:
        final_action = vlm_signal.action
        final_confidence = min(
            vlm_signal.confidence * 0.7 + sentiment.intensity * 0.3,
            0.99,
        )
    else:
        final_action = vlm_signal.action
        final_confidence = vlm_signal.confidence * 0.5

    vlm_model = (
        config.ollama_vlm_model
        if vlm_provider == "ollama"
        else Path(config.llama_cpp_model_path).name
        if vlm_provider == "llama_cpp"
        else config.vllm_model_name
    )

    return TradingDecision(
        vlm_signal=vlm_signal,
        sentiment=sentiment,
        final_action=final_action,
        final_confidence=round(final_confidence, 3),
        meta={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vlm_model": vlm_model,
            "vlm_provider": vlm_provider,
            "sentiment_model": config.ollama_model,
            "signals_aligned": vlm_signal.action == sentiment_bias,
            "platform": "Apple Silicon M4"
            if vlm_provider in ("ollama", "llama_cpp")
            else "CUDA GPU",
        },
    )


# ============================================================================
# Full Pipeline
# ============================================================================

async def run_pipeline(
    image_path: Path,
    news_text: str,
    vlm_provider: str = "llama_cpp",
    config: Any = None,
    on_status: Any = None,
) -> TradingDecision:
    """Run the full multimodal alpha-signal pipeline.

    Args:
        image_path: Path to the candlestick chart image.
        news_text: Financial news context.
        vlm_provider: VLM backend ('ollama', 'llama_cpp', 'vllm').
        config: PipelineConfig (uses default if None).
        on_status: Optional callback for UI status updates.

    Returns:
        Merged TradingDecision.
    """
    if config is None:
        from config import pipeline_cfg
        config = pipeline_cfg
        vlm_provider = config.vlm_provider

    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal Extractor — Pipeline")
    logger.info(f"  VLM Provider: {vlm_provider.upper()}")
    logger.info("=" * 70)

    vlm_task = asyncio.create_task(
        run_vlm_chain(image_path, news_text, vlm_provider, config, on_status)
    )
    sentiment_task = asyncio.create_task(
        run_sentiment_chain(news_text, config, on_status)
    )

    vlm_signal, sentiment = await asyncio.gather(vlm_task, sentiment_task)
    decision = merge_signals(vlm_signal, sentiment, vlm_provider, config)

    logger.info("=" * 70)
    logger.info(f"  FINAL: {decision.final_action.value} ({decision.final_confidence:.1%})")
    logger.info("=" * 70)

    return decision


def run_pipeline_sync(
    image_path: Path,
    news_text: str,
    vlm_provider: str = "llama_cpp",
    config: Any = None,
    on_status: Any = None,
) -> TradingDecision:
    """Synchronous wrapper for the async pipeline."""
    return asyncio.run(
        run_pipeline(image_path, news_text, vlm_provider, config, on_status)
    )
