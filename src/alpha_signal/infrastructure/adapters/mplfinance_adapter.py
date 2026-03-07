"""mplfinance_adapter.py — Chart rendering using mplfinance."""

from datetime import datetime
from pathlib import Path

import mplfinance as mpf
import pandas as pd

from alpha_signal.application.ports import ChartRendererPort
from alpha_signal.infrastructure.logger import logger


class MplfinanceAdapter(ChartRendererPort):
    """Adapter for generating technical charts as images using mplfinance."""

    def render_chart(self, df: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
        """Render a live candlestick chart with RSI and Bollinger Bands."""
        logger.info("📊 Rendering chart...")

        add_plots = [
            mpf.make_addplot(df["BB_Upper"], color="steelblue", linestyle="--", width=0.8),
            mpf.make_addplot(df["BB_Lower"], color="steelblue", linestyle="--", width=0.8),
            mpf.make_addplot(df["BB_Middle"], color="orange", linestyle="-", width=0.8),
            mpf.make_addplot(df["RSI"], panel=2, color="purple", ylabel="RSI", width=1.0),
            mpf.make_addplot(
                pd.Series(30.0, index=df.index), panel=2, color="green",
                linestyle="--", width=0.5,
            ),
            mpf.make_addplot(
                pd.Series(70.0, index=df.index), panel=2, color="red",
                linestyle="--", width=0.5,
            ),
        ]

        mc = mpf.make_marketcolors(
            up="limegreen", down="tomato", edge="inherit",
            wick="inherit", volume="steelblue",
        )
        style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="gray")

        chart_path = output_dir / f"live_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        title = (
            f"{ticker} | {df.index[0].strftime('%Y-%m-%d')} → "
            f"{df.index[-1].strftime('%Y-%m-%d')} | LIVE"
        )

        mpf.plot(
            df[["Open", "High", "Low", "Close", "Volume"]],
            type="candle",
            style=style,
            addplot=add_plots,
            volume=True,
            title=title,
            figsize=(14, 9),
            panel_ratios=(4, 1, 2),
            savefig=dict(fname=str(chart_path), dpi=120, bbox_inches="tight"),
        )

        logger.info(f"  ✓ Chart saved → {chart_path}")
        return chart_path
