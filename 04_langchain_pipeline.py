#!/usr/bin/env python3
"""
04_langchain_pipeline.py — LangChain Orchestrator for Multimodal Alpha Signals.

This module is the production-grade orchestration layer that:
  1. Connects to Ollama VLM (llama3.2-vision) OR vLLM (fine-tuned Qwen2-VL)
     for multimodal analysis (chart image + text context).
  2. Connects to Ollama (llama3:8b) for text-only sentiment extraction.
  3. Merges both signals into a final structured TradingDecision.

Apple Silicon M4 (24 GB):
  - Uses Ollama for both chains (VLM + sentiment).
  - llama3.2-vision:11b for chart analysis (~7 GB memory).
  - llama3:8b for sentiment (~5 GB memory).

CUDA Machines:
  - Uses vLLM for VLM (fine-tuned model with paged attention).
  - Uses Ollama for sentiment.

Key features:
  - Pydantic output parsing for strict JSON schema enforcement.
  - Multimodal prompts with base64 image injection.
  - Async-ready with sync wrapper for CLI usage.
  - Retry logic with exponential backoff (tenacity).
  - Comprehensive typing, docstrings, and structured logging.

Usage:
    python 04_langchain_pipeline.py --image chart.png --news "Apple beats earnings"
    python 04_langchain_pipeline.py --demo   # Uses dataset chart + sample news

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import pipeline_cfg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, pipeline_cfg.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models — Structured Output Schemas
# ============================================================================

class TradeAction(str, Enum):
    """Allowed trading actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal(BaseModel):
    """Structured trading signal output from the VLM.

    This schema is enforced via LangChain's PydanticOutputParser
    to guarantee the VLM responds with valid, parseable JSON.
    """
    action: TradeAction = Field(
        description="Trading action: BUY, SELL, or HOLD"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence score between 0.0 and 1.0"
    )
    entry_price: float = Field(
        gt=0.0,
        description="Suggested entry price for the trade"
    )
    stop_loss: float = Field(
        gt=0.0,
        description="Stop-loss price level for risk management"
    )
    take_profit: float = Field(
        gt=0.0,
        description="Take-profit price target"
    )
    reasoning: str = Field(
        min_length=10,
        description="Brief technical reasoning for the signal"
    )


class SentimentResult(BaseModel):
    """Structured sentiment analysis output from the text LLM (Ollama)."""
    sentiment: str = Field(
        description="Market sentiment: BULLISH, BEARISH, or NEUTRAL"
    )
    intensity: float = Field(
        ge=0.0, le=1.0,
        description="Sentiment intensity score (0.0 = weak, 1.0 = strong)"
    )
    key_factors: list[str] = Field(
        description="List of key factors driving the sentiment"
    )
    summary: str = Field(
        description="One-sentence summary of the sentiment analysis"
    )


class TradingDecision(BaseModel):
    """Final merged trading decision combining VLM and sentiment signals."""
    vlm_signal: TradingSignal = Field(
        description="Trading signal from the Vision-Language Model"
    )
    sentiment: SentimentResult = Field(
        description="Sentiment analysis from the text LLM"
    )
    final_action: TradeAction = Field(
        description="Final recommended action after merging both signals"
    )
    final_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Merged confidence score"
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (timestamps, model versions, etc.)"
    )


# ============================================================================
# Image Utilities
# ============================================================================

def encode_image_to_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string.

    Args:
        image_path: Path to the image file (PNG or JPEG).

    Returns:
        Base64-encoded string (raw, without data URI prefix for Ollama).

    Raises:
        FileNotFoundError: If the image does not exist.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_as_data_uri(image_path: Path) -> str:
    """Read an image and return a data URI for OpenAI-compatible APIs.

    Args:
        image_path: Path to the image file.

    Returns:
        Data URI string (e.g., 'data:image/png;base64,...').
    """
    path = Path(image_path).resolve()
    suffix = path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(suffix, "image/png")
    b64 = encode_image_to_base64(path)
    return f"data:{mime_type};base64,{b64}"


# ============================================================================
# Chain 1: VLM Chain — Vision + Text → TradingSignal
# ============================================================================

def _build_vlm_system_prompt(format_instructions: str) -> str:
    """Build the system prompt for the VLM chain.

    Args:
        format_instructions: Pydantic parser format instructions.

    Returns:
        System prompt string.
    """
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
) -> TradingSignal:
    """Execute the VLM chain with a chart image and news context.

    Automatically selects the backend:
      - Ollama (llama3.2-vision) on Apple Silicon M4
      - vLLM (fine-tuned Qwen2-VL) on CUDA machines

    Args:
        image_path: Path to the candlestick chart image.
        news_text: Financial news context string.

    Returns:
        Parsed TradingSignal object.
    """
    parser = PydanticOutputParser(pydantic_object=TradingSignal)
    format_instructions = parser.get_format_instructions()
    system_prompt = _build_vlm_system_prompt(format_instructions)

    user_text = (
        f"Analyse le graphique financier ci-dessus et le contexte "
        f"d'actualité suivant :\n\n"
        f"**Actualité** : {news_text}\n\n"
        f"Génère un signal de trading structuré au format JSON demandé."
    )

    if pipeline_cfg.vlm_provider == "ollama":
        # ---- OLLAMA BACKEND (Apple Silicon M4) ----
        logger.info(f"Invoking VLM via Ollama ({pipeline_cfg.ollama_vlm_model})...")

        llm = ChatOllama(
            base_url=pipeline_cfg.ollama_base_url,
            model=pipeline_cfg.ollama_vlm_model,
            temperature=pipeline_cfg.vlm_temperature,
        )

        # For Ollama vision models, images are passed as base64 in the
        # message content using the `image_url` format
        img_b64 = encode_image_to_base64(image_path)
        img_data_uri = encode_image_as_data_uri(image_path)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": img_data_uri},
                },
                {"type": "text", "text": user_text},
            ]),
        ]

        response = await llm.ainvoke(messages)
        raw_content = response.content

    elif pipeline_cfg.vlm_provider == "llama_cpp":
        # ---- LLAMA.CPP DIRECT GGUF BACKEND (fine-tuned model) ----
        from llama_cpp import Llama

        gguf_path = pipeline_cfg.llama_cpp_model_path
        logger.info(f"Invoking VLM via llama.cpp ({Path(gguf_path).name})...")

        # Load model (cached as module-level singleton for performance)
        global _llama_model
        if "_llama_model" not in globals() or _llama_model is None:
            logger.info("  Loading GGUF model (first call, may take ~10s)...")
            _llama_model = Llama(
                model_path=gguf_path,
                n_gpu_layers=pipeline_cfg.llama_cpp_n_gpu_layers,
                n_ctx=pipeline_cfg.llama_cpp_n_ctx,
                verbose=False,
            )
            logger.info("  ✓ Model loaded")

        # Build prompt for text-only inference (GGUF text model)
        img_b64 = encode_image_to_base64(image_path)
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        result = _llama_model(
            full_prompt,
            max_tokens=pipeline_cfg.vlm_max_tokens,
            temperature=pipeline_cfg.vlm_temperature,
            stop=["<|im_end|>"],
        )
        raw_content = result["choices"][0]["text"]

    else:
        # ---- vLLM BACKEND (CUDA machines) ----
        from langchain_openai import ChatOpenAI

        logger.info(f"Invoking VLM via vLLM ({pipeline_cfg.vllm_model_name})...")

        llm = ChatOpenAI(
            base_url=pipeline_cfg.vllm_base_url,
            api_key=pipeline_cfg.vllm_api_key,
            model=pipeline_cfg.vllm_model_name,
            temperature=pipeline_cfg.vlm_temperature,
            max_tokens=pipeline_cfg.vlm_max_tokens,
        )

        img_data_uri = encode_image_as_data_uri(image_path)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": img_data_uri, "detail": "high"},
                },
                {"type": "text", "text": user_text},
            ]),
        ]

        response = await llm.ainvoke(messages)
        raw_content = response.content

    # Parse the response
    logger.debug(f"Raw VLM response: {raw_content[:300]}...")

    # Try to extract JSON from response (model might wrap it in markdown)
    signal = _parse_json_response(raw_content, parser)
    logger.info(f"VLM Signal: {signal.action.value} (confidence: {signal.confidence:.2f})")
    return signal


def _parse_json_response(
    raw_content: str,
    parser: PydanticOutputParser,
) -> TradingSignal:
    """Parse JSON from model response, handling common formatting issues.

    Models sometimes wrap JSON in markdown code blocks or add extra text.
    This function handles those cases gracefully.

    Args:
        raw_content: Raw string response from the model.
        parser: Pydantic output parser.

    Returns:
        Parsed TradingSignal.
    """
    content = raw_content.strip()

    # Try direct parse first
    try:
        return parser.parse(content)
    except Exception:
        pass

    # Try extracting JSON from markdown code block
    if "```json" in content:
        json_block = content.split("```json")[1].split("```")[0].strip()
        try:
            return parser.parse(json_block)
        except Exception:
            pass

    if "```" in content:
        json_block = content.split("```")[1].split("```")[0].strip()
        try:
            return parser.parse(json_block)
        except Exception:
            pass

    # Try finding JSON object in the string
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = content[start:end]
        try:
            return parser.parse(json_str)
        except Exception:
            pass

    # Last resort: raise with helpful message
    raise ValueError(
        f"Could not parse VLM response as TradingSignal JSON. "
        f"Raw response (first 500 chars): {content[:500]}"
    )


# ============================================================================
# Chain 2: Sentiment Chain (Ollama — Text only → SentimentResult)
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        f"Sentiment call failed, retrying ({retry_state.attempt_number}/3)..."
    ),
)
async def run_sentiment_chain(news_text: str) -> SentimentResult:
    """Execute the sentiment chain on financial news via Ollama.

    Args:
        news_text: Financial news text to analyze.

    Returns:
        Parsed SentimentResult object.
    """
    parser = PydanticOutputParser(pydantic_object=SentimentResult)
    format_instructions = parser.get_format_instructions()

    llm = ChatOllama(
        base_url=pipeline_cfg.ollama_base_url,
        model=pipeline_cfg.ollama_model,
        temperature=pipeline_cfg.ollama_temperature,
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

    logger.info(f"Invoking Sentiment chain (Ollama {pipeline_cfg.ollama_model})...")
    response = await chain.ainvoke({
        "format_instructions": format_instructions,
        "news_text": news_text,
    })

    # Parse with fallback
    content = response.content.strip()
    try:
        sentiment = parser.parse(content)
    except Exception:
        # Try extracting JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            sentiment = parser.parse(content[start:end])
        else:
            raise

    logger.info(
        f"Sentiment: {sentiment.sentiment} "
        f"(intensity: {sentiment.intensity:.2f})"
    )
    return sentiment


# ============================================================================
# Signal Merger
# ============================================================================

def merge_signals(
    vlm_signal: TradingSignal,
    sentiment: SentimentResult,
) -> TradingDecision:
    """Merge VLM trading signal with text sentiment into a final decision.

    Merging logic:
      - If both signals agree → boost confidence.
      - If signals conflict → reduce confidence, VLM takes priority.
      - HOLD from VLM is respected regardless of sentiment.

    Args:
        vlm_signal: Output from the VLM chain.
        sentiment: Output from the sentiment chain.

    Returns:
        Unified TradingDecision.
    """
    from datetime import datetime, timezone

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
        pipeline_cfg.ollama_vlm_model
        if pipeline_cfg.vlm_provider == "ollama"
        else Path(pipeline_cfg.llama_cpp_model_path).name
        if pipeline_cfg.vlm_provider == "llama_cpp"
        else pipeline_cfg.vllm_model_name
    )

    return TradingDecision(
        vlm_signal=vlm_signal,
        sentiment=sentiment,
        final_action=final_action,
        final_confidence=round(final_confidence, 3),
        meta={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vlm_model": vlm_model,
            "vlm_provider": pipeline_cfg.vlm_provider,
            "sentiment_model": pipeline_cfg.ollama_model,
            "signals_aligned": vlm_signal.action == sentiment_bias,
            "platform": "Apple Silicon M4"
            if pipeline_cfg.vlm_provider in ("ollama", "llama_cpp")
            else "CUDA GPU",
        },
    )


# ============================================================================
# Main Pipeline
# ============================================================================

async def run_pipeline(
    image_path: Path,
    news_text: str,
) -> TradingDecision:
    """Run the full multimodal alpha-signal pipeline.

    Executes both chains concurrently, then merges results.

    Args:
        image_path: Path to the candlestick chart image.
        news_text: Financial news context.

    Returns:
        Merged TradingDecision.
    """
    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal Extractor — Pipeline")
    logger.info(f"  VLM Provider: {pipeline_cfg.vlm_provider.upper()}")
    logger.info("=" * 70)
    logger.info(f"  Image : {image_path}")
    logger.info(f"  News  : {news_text[:80]}...")
    logger.info("=" * 70)

    # Run both chains concurrently
    vlm_task = asyncio.create_task(run_vlm_chain(image_path, news_text))
    sentiment_task = asyncio.create_task(run_sentiment_chain(news_text))

    vlm_signal, sentiment = await asyncio.gather(vlm_task, sentiment_task)

    decision = merge_signals(vlm_signal, sentiment)

    logger.info("=" * 70)
    logger.info("  FINAL DECISION")
    logger.info(f"  Action     : {decision.final_action.value}")
    logger.info(f"  Confidence : {decision.final_confidence:.1%}")
    logger.info(f"  Stop Loss  : {decision.vlm_signal.stop_loss}")
    logger.info(f"  Take Profit: {decision.vlm_signal.take_profit}")
    logger.info(f"  Aligned    : {decision.meta.get('signals_aligned')}")
    logger.info("=" * 70)

    return decision


def run_pipeline_sync(image_path: Path, news_text: str) -> TradingDecision:
    """Synchronous wrapper for the async pipeline."""
    return asyncio.run(run_pipeline(image_path, news_text))


# ============================================================================
# Demo Mode
# ============================================================================

def run_demo() -> TradingDecision:
    """Run the pipeline with a sample chart from the generated dataset.

    Extracts the first chart from the JSONL dataset and uses a sample news.

    Returns:
        TradingDecision from the demo run.
    """
    dataset_dir = Path(__file__).parent / "dataset"
    jsonl_path = dataset_dir / "training_data.jsonl"

    if not jsonl_path.exists():
        logger.error("Dataset not found. Run 01_generate_dataset.py first.")
        sys.exit(1)

    # Extract first chart image from JSONL
    with open(jsonl_path, "r") as f:
        first_sample = json.loads(f.readline())

    for block in first_sample["messages"][1]["content"]:
        if block.get("type") == "image":
            img_data = block["image"].split(",", 1)[1]
            img_bytes = base64.b64decode(img_data)
            test_img = dataset_dir / "demo_chart.png"
            test_img.write_bytes(img_bytes)
            logger.info(f"Extracted demo chart → {test_img}")
            break
    else:
        logger.error("No image found in first JSONL sample.")
        sys.exit(1)

    sample_news = (
        "Apple Inc. vient d'annoncer des résultats trimestriels record "
        "avec une hausse de 12% du chiffre d'affaires. Les analystes de "
        "Goldman Sachs relèvent leur objectif de cours de 15%. Le segment "
        "Services atteint un nouveau sommet historique."
    )

    return run_pipeline_sync(test_img, sample_news)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Multimodal Alpha-Signal Extractor — LangChain Pipeline"
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Path to the candlestick chart image",
    )
    parser.add_argument(
        "--news", "-n", type=str, default=None,
        help="Financial news text to analyze",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo mode with a sample chart from the dataset",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional path to save the JSON output",
    )

    args = parser.parse_args()

    try:
        if args.demo:
            decision = run_demo()
        elif args.image and args.news:
            decision = run_pipeline_sync(Path(args.image), args.news)
        else:
            parser.error("Either --demo or both --image and --news are required")
            return

        output_json = decision.model_dump_json(indent=2)
        print("\n" + output_json)

        if args.output:
            Path(args.output).write_text(output_json, encoding="utf-8")
            logger.info(f"Decision saved to {args.output}")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
