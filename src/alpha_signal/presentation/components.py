"""components.py — Shared presentation components (Plotly).

Consolidates UI components like interactive charts for the Streamlit app.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_plotly_chart(
    df: pd.DataFrame,
    ticker: str = "",
    height: int = 700,
) -> go.Figure:
    """Create an interactive Plotly chart with Candlestick, Bollinger Bands, RSI, and Volume.

    Args:
        df: DataFrame with OHLCV + RSI + BB_Upper + BB_Lower + BB_Middle.
        ticker: Ticker symbol for chart title.
        height: Total chart height in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25],
        subplot_titles=(f"{ticker} — Candlestick + Bollinger Bands", "Volume", "RSI"),
    )

    # --- Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4757",
            increasing_fillcolor="#00d4aa",
            decreasing_fillcolor="#ff4757",
        ),
        row=1, col=1,
    )

    # --- Bollinger Bands ---
    if "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(99,179,237,0.6)", width=1, dash="dash"),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="rgba(99,179,237,0.6)", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(99,179,237,0.08)",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Middle"],
                name="BB Mid (SMA)",
                line=dict(color="#f7b731", width=1),
                showlegend=False,
            ),
            row=1, col=1,
        )

    # --- Volume ---
    colors = [
        "#00d4aa" if c >= o else "#ff4757"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- RSI ---
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["RSI"],
                name="RSI",
                line=dict(color="#a55eea", width=1.5),
                showlegend=False,
            ),
            row=3, col=1,
        )
        # Reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4757", line_width=0.8, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", line_width=0.8, row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.05)", line_width=0, row=3, col=1)

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=60, r=30, t=50, b=30),
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Axis styling
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
            row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
            row=i, col=1,
        )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    return fig
