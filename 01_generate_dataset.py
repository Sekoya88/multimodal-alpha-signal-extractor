#!/usr/bin/env python3
"""
01_generate_dataset.py — Synthetic Multimodal Dataset Generator.

Downloads AAPL historical data, computes technical indicators (RSI, Bollinger
Bands), renders candlestick charts via mplfinance, and produces an
Unsloth-compatible JSONL training file pairing each chart (base64-encoded)
with a synthetic news snippet and a ground-truth trading signal.

Usage:
    python 01_generate_dataset.py

Output:
    dataset/training_data.jsonl

Author: Nicolas
License: MIT
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image

from config import CHARTS_DIR, DATASET_DIR, dataset_cfg

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
# Synthetic News Templates
# ---------------------------------------------------------------------------
BULLISH_NEWS: list[str] = [
    "Apple Inc. vient d'annoncer des résultats trimestriels supérieurs aux attentes "
    "avec une hausse de {pct}% du chiffre d'affaires. Les analystes relèvent leurs "
    "objectifs de cours.",
    "Le segment Services d'Apple atteint un nouveau record de revenus. "
    "Les marges opérationnelles s'améliorent de {pct} points de base.",
    "Goldman Sachs relève sa recommandation sur AAPL à « Surpondérer » avec un "
    "objectif de cours en hausse de {pct}%. Le momentum institutionnel est fort.",
    "Apple annonce un programme de rachat d'actions de {amount}Md$ et augmente "
    "son dividende de {pct}%. Signal positif pour les actionnaires.",
    "Les ventes d'iPhone en Chine dépassent les prévisions de {pct}%, "
    "dissipant les craintes de ralentissement géopolitique.",
]

BEARISH_NEWS: list[str] = [
    "Les ventes d'Apple en Chine chutent de {pct}% au dernier trimestre, "
    "affectées par la concurrence locale de Huawei et la pression réglementaire.",
    "La Fed annonce un maintien des taux élevés plus longtemps que prévu. "
    "Les valeurs technologiques, dont AAPL, subissent des pressions vendeuses.",
    "Des rapports indiquent un ralentissement de la demande d'iPhone {pct}% "
    "en dessous des estimations. Les stocks chez les distributeurs augmentent.",
    "La Commission européenne impose une amende de {amount}Md€ à Apple pour "
    "pratiques anticoncurrentielles sur l'App Store.",
    "Le rendement du Treasury 10 ans dépasse {pct}%, provoquant une rotation "
    "sectorielle massive hors des valeurs de croissance.",
]

NEUTRAL_NEWS: list[str] = [
    "Apple présente des résultats en ligne avec les attentes du consensus. "
    "Le cours reste stable avec un volume d'échanges modéré.",
    "Le marché évolue sans direction claire aujourd'hui. Les indices oscillent "
    "autour de l'équilibre en attendant les prochains catalyseurs macro.",
    "L'action AAPL consolide dans un range étroit de {pct}% cette semaine. "
    "Les analystes maintiennent leurs objectifs inchangés.",
]

# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        series: Price series (typically 'Close').
        period: Look-back window for RSI computation.

    Returns:
        RSI values as a pandas Series (0–100 scale).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (middle, upper, lower).

    Args:
        series: Price series (typically 'Close').
        period: SMA window length.
        num_std: Number of standard deviations for the bands.

    Returns:
        Tuple of (middle_band, upper_band, lower_band).
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


# ---------------------------------------------------------------------------
# Chart Rendering
# ---------------------------------------------------------------------------

def render_candlestick_chart(
    df_window: pd.DataFrame,
    rsi: pd.Series,
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    bb_middle: pd.Series,
    title: str = "",
) -> bytes:
    """Render a candlestick chart with Bollinger Bands and RSI sub-plot.

    The chart is returned as PNG bytes (not saved to disk).

    Args:
        df_window: OHLCV DataFrame slice for the current window.
        rsi: RSI values aligned to df_window index.
        bb_upper: Upper Bollinger Band.
        bb_lower: Lower Bollinger Band.
        bb_middle: Middle Bollinger Band (SMA).
        title: Chart title.

    Returns:
        PNG image bytes.
    """
    # Bollinger Bands as additional plots on main panel
    add_plots = [
        mpf.make_addplot(bb_upper, color="steelblue", linestyle="--", width=0.8),
        mpf.make_addplot(bb_lower, color="steelblue", linestyle="--", width=0.8),
        mpf.make_addplot(bb_middle, color="orange", linestyle="-", width=0.8),
        # RSI on secondary panel
        mpf.make_addplot(rsi, panel=2, color="purple", ylabel="RSI", width=1.0),
    ]

    # Horizontal RSI reference lines via secondary panel
    rsi_30 = pd.Series(30.0, index=df_window.index)
    rsi_70 = pd.Series(70.0, index=df_window.index)
    add_plots.append(
        mpf.make_addplot(rsi_30, panel=2, color="green", linestyle="--", width=0.5)
    )
    add_plots.append(
        mpf.make_addplot(rsi_70, panel=2, color="red", linestyle="--", width=0.5)
    )

    # Custom style
    mc = mpf.make_marketcolors(
        up="limegreen", down="tomato", edge="inherit",
        wick="inherit", volume="steelblue",
    )
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="gray")

    buf = io.BytesIO()
    mpf.plot(
        df_window,
        type="candle",
        style=style,
        addplot=add_plots,
        volume=True,
        title=title,
        figsize=(12, 8),
        panel_ratios=(4, 1, 2),
        savefig=dict(fname=buf, dpi=dataset_cfg.chart_dpi, bbox_inches="tight"),
    )
    buf.seek(0)
    return buf.read()


def image_bytes_to_base64(img_bytes: bytes) -> str:
    """Encode raw image bytes to a base64 string.

    Args:
        img_bytes: PNG or JPEG image data.

    Returns:
        Base64-encoded string representation of the image.
    """
    return base64.b64encode(img_bytes).encode() if isinstance(img_bytes, bytes) else ""


# ---------------------------------------------------------------------------
# Label Generation
# ---------------------------------------------------------------------------

def generate_label(
    df: pd.DataFrame,
    window_end_idx: int,
    rsi_value: float,
    close_price: float,
    bb_upper_value: float,
    bb_lower_value: float,
    forward_days: int = 5,
) -> dict[str, Any]:
    """Determine ground-truth trading signal from forward returns + indicators.

    Labeling heuristic:
      - BUY  if RSI < 35 and price near/below lower Bollinger Band and
        forward return > +1%.
      - SELL if RSI > 65 and price near/above upper Bollinger Band and
        forward return < -1%.
      - HOLD otherwise.

    Confidence is derived from the normalized distance of RSI from neutral (50)
    and the forward return magnitude.

    Args:
        df: Full OHLCV DataFrame.
        window_end_idx: Integer location of the last bar in the window.
        rsi_value: Current RSI reading.
        close_price: Current close price.
        bb_upper_value: Upper Bollinger Band value.
        bb_lower_value: Lower Bollinger Band value.
        forward_days: Number of days to look ahead for return calculation.

    Returns:
        Dict with keys: signal, confidence, entry_price, stop_loss, take_profit.
    """
    # Forward return
    future_idx = min(window_end_idx + forward_days, len(df) - 1)
    future_price = df.iloc[future_idx]["Close"]
    forward_return = (future_price - close_price) / close_price

    # Confidence from RSI divergence (0.5–0.95 scale)
    rsi_divergence = abs(rsi_value - 50.0) / 50.0
    confidence = round(min(0.5 + rsi_divergence * 0.5, 0.95), 2)

    # Signal logic
    if rsi_value < 45 and close_price <= bb_lower_value * 1.05 and forward_return > 0.003:
        signal = "BUY"
        stop_loss = round(close_price * 0.97, 2)
        take_profit = round(close_price * 1.05, 2)
    elif rsi_value > 55 and close_price >= bb_upper_value * 0.95 and forward_return < -0.003:
        signal = "SELL"
        stop_loss = round(close_price * 1.03, 2)
        take_profit = round(close_price * 0.95, 2)
    else:
        signal = "HOLD"
        stop_loss = round(close_price * 0.98, 2)
        take_profit = round(close_price * 1.02, 2)
        confidence = round(confidence * 0.6, 2)  # Lower confidence for HOLD

    return {
        "signal": signal,
        "confidence": confidence,
        "entry_price": round(close_price, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


# ---------------------------------------------------------------------------
# News Generation
# ---------------------------------------------------------------------------

def generate_synthetic_news(signal: str) -> str:
    """Generate a contextual synthetic news snippet matching the signal.

    Args:
        signal: One of "BUY", "SELL", or "HOLD".

    Returns:
        A realistic-sounding financial news string.
    """
    pct = round(random.uniform(2.0, 15.0), 1)
    amount = round(random.uniform(5.0, 100.0), 0)

    if signal == "BUY":
        template = random.choice(BULLISH_NEWS)
    elif signal == "SELL":
        template = random.choice(BEARISH_NEWS)
    else:
        template = random.choice(NEUTRAL_NEWS)

    return template.format(pct=pct, amount=int(amount))


# ---------------------------------------------------------------------------
# JSONL Formatting (Unsloth Conversational VLM Format)
# ---------------------------------------------------------------------------

def format_training_sample(
    image_b64: str,
    news_text: str,
    label: dict[str, Any],
) -> dict[str, Any]:
    """Format a single training sample in Unsloth conversational VLM format.

    The format follows the chat-template structure expected by Unsloth's
    `FastVisionModel` with `messages` containing `image` and `text` blocks.

    Args:
        image_b64: Base64-encoded PNG chart image.
        news_text: Synthetic financial news snippet.
        label: Ground-truth signal dictionary.

    Returns:
        A dict ready for JSON serialization into a JSONL line.
    """
    system_prompt = (
        "Tu es un analyste quantitatif senior dans un hedge fund de premier plan. "
        "Tu analyses des graphiques financiers en chandeliers (avec RSI et bandes "
        "de Bollinger) ainsi que le contexte d'actualité de marché pour générer "
        "des signaux de trading structurés. Réponds toujours en JSON strict."
    )

    user_content = (
        f"Analyse le graphique financier ci-joint et le contexte d'actualité "
        f"suivant :\n\n**Actualité** : {news_text}\n\n"
        f"Génère un signal de trading structuré au format JSON avec les champs : "
        f"action (BUY/SELL/HOLD), confidence (0.0-1.0), entry_price, stop_loss, "
        f"take_profit, et reasoning (explication courte)."
    )

    assistant_response = json.dumps(
        {
            "action": label["signal"],
            "confidence": label["confidence"],
            "entry_price": label["entry_price"],
            "stop_loss": label["stop_loss"],
            "take_profit": label["take_profit"],
            "reasoning": _generate_reasoning(label),
        },
        ensure_ascii=False,
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/png;base64,{image_b64}",
                    },
                    {"type": "text", "text": user_content},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            },
        ],
    }


def _generate_reasoning(label: dict[str, Any]) -> str:
    """Generate a short reasoning string for the training label.

    Args:
        label: Ground-truth signal dictionary.

    Returns:
        Human-readable reasoning string.
    """
    signal = label["signal"]
    conf = label["confidence"]

    if signal == "BUY":
        return (
            f"RSI en zone de survente, prix proche de la bande de Bollinger "
            f"inférieure. Divergence haussière détectée. Confiance {conf:.0%}."
        )
    elif signal == "SELL":
        return (
            f"RSI en zone de surachat, prix proche de la bande de Bollinger "
            f"supérieure. Pression vendeuse anticipée. Confiance {conf:.0%}."
        )
    else:
        return (
            f"Indicateurs en zone neutre, pas de signal directionnel clair. "
            f"Recommandation de maintien de position. Confiance {conf:.0%}."
        )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full synthetic dataset generation pipeline."""
    logger.info("=" * 70)
    logger.info("  Multimodal Alpha-Signal Extractor — Dataset Generator")
    logger.info("=" * 70)

    # Ensure directories exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download market data
    # ------------------------------------------------------------------
    logger.info(f"Downloading {dataset_cfg.ticker} data (period={dataset_cfg.period})...")
    ticker = yf.Ticker(dataset_cfg.ticker)
    df = ticker.history(period=dataset_cfg.period, interval=dataset_cfg.interval)

    if df.empty:
        logger.error("No data returned from yfinance. Check ticker/period.")
        sys.exit(1)

    # Ensure DatetimeIndex for mplfinance
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    logger.info(f"Downloaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # ------------------------------------------------------------------
    # 2. Compute indicators on full series
    # ------------------------------------------------------------------
    logger.info("Computing RSI and Bollinger Bands...")
    df["RSI"] = compute_rsi(df["Close"], period=dataset_cfg.rsi_period)
    bb_mid, bb_up, bb_low = compute_bollinger_bands(
        df["Close"],
        period=dataset_cfg.bollinger_period,
        num_std=dataset_cfg.bollinger_std,
    )
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_up
    df["BB_Lower"] = bb_low

    # Drop warmup NaNs
    warmup = max(dataset_cfg.rsi_period, dataset_cfg.bollinger_period)
    df = df.iloc[warmup:].copy()
    logger.info(f"After warmup removal: {len(df)} bars available")

    # ------------------------------------------------------------------
    # 3. Sliding window → generate samples
    # ------------------------------------------------------------------
    samples: list[dict[str, Any]] = []
    window = dataset_cfg.window_size
    stride = dataset_cfg.stride

    total_windows = (len(df) - window - dataset_cfg.forward_return_days) // stride
    logger.info(f"Generating {total_windows} samples (window={window}, stride={stride})...")

    for i in range(0, len(df) - window - dataset_cfg.forward_return_days, stride):
        df_window = df.iloc[i : i + window]

        # Current bar values (last bar in window)
        last_idx = i + window - 1
        last_row = df.iloc[last_idx]
        rsi_val = last_row["RSI"]
        close_val = last_row["Close"]
        bb_up_val = last_row["BB_Upper"]
        bb_low_val = last_row["BB_Lower"]

        # Generate label
        label = generate_label(
            df, last_idx, rsi_val, close_val, bb_up_val, bb_low_val,
            forward_days=dataset_cfg.forward_return_days,
        )

        # Generate news
        news = generate_synthetic_news(label["signal"])

        # Render chart
        chart_title = (
            f"{dataset_cfg.ticker} | "
            f"{df_window.index[0].strftime('%Y-%m-%d')} → "
            f"{df_window.index[-1].strftime('%Y-%m-%d')}"
        )
        try:
            img_bytes = render_candlestick_chart(
                df_window=df_window[["Open", "High", "Low", "Close", "Volume"]],
                rsi=df_window["RSI"],
                bb_upper=df_window["BB_Upper"],
                bb_lower=df_window["BB_Lower"],
                bb_middle=df_window["BB_Middle"],
                title=chart_title,
            )
        except Exception as e:
            logger.warning(f"Chart rendering failed for window {i}: {e}")
            continue

        # Encode image
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Format sample
        sample = format_training_sample(img_b64, news, label)
        samples.append(sample)

        if (len(samples)) % 10 == 0:
            logger.info(f"  Generated {len(samples)}/{total_windows} samples...")

    # ------------------------------------------------------------------
    # 4. Write JSONL
    # ------------------------------------------------------------------
    output_path = dataset_cfg.output_jsonl
    logger.info(f"Writing {len(samples)} samples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Stats
    signal_counts = {}
    for s in samples:
        msg = s["messages"][-1]["content"][0]["text"]
        action = json.loads(msg)["action"]
        signal_counts[action] = signal_counts.get(action, 0) + 1

    logger.info(f"Dataset written: {output_path}")
    logger.info(f"  Total samples : {len(samples)}")
    logger.info(f"  Signal distrib: {signal_counts}")
    logger.info(f"  File size     : {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("Done ✓")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
