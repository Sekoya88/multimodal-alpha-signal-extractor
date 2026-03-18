"""
Streamlit App — Alpha-Signal Extractor

Premium Cyber-Fintech / Glassmorphism Interface
Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Project imports ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpha_signal.presentation.components import create_plotly_chart
from alpha_signal.presentation.di_container import build_analyze_market_usecase
from alpha_signal.infrastructure.adapters.yfinance_adapter import YFinanceAdapter


# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="Alpha-Signal Extractor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CSS — Premium Cyber-Fintech & Glassmorphism
# ============================================================================

st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Theme */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1625 0%, #0d0e15 40%, #050608 100%);
        font-family: 'Inter', -apple-system, sans-serif;
        color: #e2e8f0;
    }
    
    /* Hide default Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"], section[data-testid="stSidebar"] { 
        display: none !important; 
    }

    /* Typography */
    h1, h2, h3, .title-font {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: rgba(13, 14, 21, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .nav-logo {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #050608;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
    }
    .nav-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .nav-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        letter-spacing: 0.5px;
        margin-top: -2px;
    }

    /* Glassmorphism Container / Cards */
    .glass-card {
        background: rgba(15, 17, 26, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 242, 254, 0.05);
    }

    /* Command Center (Floating Panel) */
    .command-center {
        background: rgba(20, 22, 34, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    
    /* Input Overrides */
    .stSelectbox > div > div, 
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        transition: all 0.2s;
    }
    .stSelectbox > div > div:hover, 
    .stTextInput > div > div > input:hover {
        border-color: rgba(0, 242, 254, 0.4) !important;
        background: rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Dropdown text color fix */
    .stSelectbox * {
        color: #f8fafc !important;
    }

    /* Primary CTA Button - Run Analysis */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #050608 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.6) !important;
    }

    /* Secondary CTA Button - Fetch Data */
    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #fff !important;
    }

    /* Feature Grid Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: rgba(15, 17, 26, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .feature-card:hover::before { opacity: 1; }
    
    .feature-icon-wrapper {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
    }
    .icon-cyan { background: rgba(0, 242, 254, 0.1); border: 1px solid rgba(0, 242, 254, 0.2); color: #00f2fe; }
    .icon-purple { background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.2); color: #a855f7; }
    .icon-emerald { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); color: #10b981; }
    
    .feature-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* KPIs */
    .kpi-container {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-box {
        flex: 1;
        background: rgba(15, 17, 26, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Decision Cards */
    .decision-panel {
        background: rgba(15, 17, 26, 0.6);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .decision-BUY { border: 1px solid rgba(16, 185, 129, 0.3); box-shadow: 0 0 30px rgba(16, 185, 129, 0.1); }
    .decision-SELL { border: 1px solid rgba(244, 63, 94, 0.3); box-shadow: 0 0 30px rgba(244, 63, 94, 0.1); }
    .decision-HOLD { border: 1px solid rgba(245, 158, 11, 0.3); box-shadow: 0 0 30px rgba(245, 158, 11, 0.1); }
    
    .badge-BUY { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-SELL { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-HOLD { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* News */
    .news-card {
        padding: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .news-card:last-child { border-bottom: none; }
    .news-pub { font-size: 0.7rem; color: #00f2fe; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 4px; }
    .news-title { font-size: 0.95rem; color: #f8fafc; font-weight: 500; line-height: 1.4; }
    .news-dist { font-size: 0.8rem; color: #64748b; margin-top: 6px; }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    }

    /* Labels */
    .st-emotion-cache-16idsys p {
        color: #94a3b8;
        font-size: 0.85rem;
    }
</style>
""")

# ============================================================================
# Session State
# ============================================================================

if "decision" not in st.session_state:
    st.session_state.decision = None
if "df" not in st.session_state:
    st.session_state.df = None
if "news_articles" not in st.session_state:
    st.session_state.news_articles = None

# ============================================================================
# Top Navigation
# ============================================================================

st.html("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-logo">α</div>
        <div>
            <div class="nav-title">Alpha-Signal Extractor</div>
            <div class="nav-subtitle">Multimodal Trading Intelligence</div>
        </div>
    </div>
    <div style="font-family:'JetBrains Mono'; font-size:0.8rem; color:#64748b; background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05);">
        STATUS: <span style="color:#10b981;">ONLINE</span>
    </div>
</div>
""")

# ============================================================================
# Command Center (Top Bar)
# ============================================================================

st.html('<div class="command-center">')

col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1.5, 1.5, 1.5], gap="medium")

with col1:
    ASSET_GROUPS = {
        "Magnificent 7": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
        "Semiconductors": ["AMD", "TSM", "AVGO", "INTC", "QCOM", "ARM"],
        "Finance & Fintech": ["JPM", "V", "MA", "BAC", "PYPL", "SQ", "COIN", "HOOD"],
        "Indices & ETFs": ["SPY", "QQQ", "DIA", "IWM", "VIX", "ARKK"],
        "Healthcare": ["LLY", "UNH", "JNJ", "ABBV", "NVO"],
        "Energy & Industrials": ["XOM", "CVX", "CAT", "GE", "BA", "LMT"],
        "Consumer": ["WMT", "PG", "KO", "PEP", "COST", "MCD", "DIS"]
    }
    FLAT_ASSETS = [f"{t}  —  {category}" for category, tickers in ASSET_GROUPS.items() for t in tickers]

    selected_asset = st.selectbox(
        "Market Asset",
        FLAT_ASSETS,
        index=0,
    )
    ticker = selected_asset.split("  —  ")[0]

with col2:
    days = st.number_input("Window", min_value=10, max_value=200, value=60, step=10)

with col3:
    # Since we fine-tuned for llama.cpp, we default to it and disable the others visually
    vlm_provider = st.selectbox(
        "VLM Engine",
        ["llama_cpp (Fine-tuned Qwen2.5)"],
        index=0,
        disabled=True
    )
    # The backend will still read "llama_cpp" from config, we just lock the UI

with col4:
    st.html("<div style='margin-top:28px;'></div>")
    fetch_only = st.button("Fetch Market Data", use_container_width=True)

with col5:
    st.html("<div style='margin-top:28px;'></div>")
    run_analysis = st.button("▶ RUN ANALYSIS", type="primary", use_container_width=True)

st.html('</div>')

# ============================================================================
# Data Fetching
# ============================================================================

if fetch_only or run_analysis:
    with st.spinner("Synchronizing market data..."):
        try:
            # For purely fetching presentation data without running pipeline
            adapter = YFinanceAdapter()
            st.session_state.df = adapter.fetch_data(ticker, days)
            st.session_state.news_articles = adapter.fetch_news_articles(ticker, max_articles=8)
        except Exception as e:
            st.error(f"Systems error: {e}")
            from alpha_signal.infrastructure.logger import logger
            logger.exception("❌ Error during market data / news synchronization:")
            st.stop()

# ============================================================================
# Main Content
# ============================================================================

if st.session_state.df is None:
    # ── Landing State / Feature Grid ──
    st.html('<div style="margin-top:4rem;"></div>')
    st.html("""
<div style="text-align:center; max-width: 800px; margin: 0 auto;">
    <h1 class="title-font" style="font-size:3rem; margin-bottom:1rem; background:linear-gradient(to right, #fff, #94a3b8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Next-Gen Algorithmic Discovery
    </h1>
    <p style="color:#94a3b8; font-size:1.1rem; line-height:1.6;">
        Deploy fine-tuned Vision-Language Models directly on raw candlestick charts, 
        merged with real-time NLP sentiment analysis to generate actionable alpha.
    </p>
</div>
""")

    st.html("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon-wrapper icon-cyan">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
        </div>
        <div class="feature-title">Computer Vision Analysis</div>
        <div class="feature-desc">Fine-tuned Qwen2.5-VL processes candlestick charts visually, analyzing price action, Bollinger Bands edge interactions, and RSI divergences identically to a human trader.</div>
    </div>
    
    <div class="feature-card">
        <div class="feature-icon-wrapper icon-purple">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
        </div>
        <div class="feature-title">LLM NLP Sentiment</div>
        <div class="feature-desc">Local LLaMA 3 instances parse real-time financial news flows via Yahoo Finance, extracting bullish/bearish catalysts and scoring market sentiment intensity.</div>
    </div>
    
    <div class="feature-card">
        <div class="feature-icon-wrapper icon-emerald">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
        </div>
        <div class="feature-title">Agentic Signal Merging</div>
        <div class="feature-desc">Clean Architecture orchestrates execution, merging purely technical visual signals with macroeconomic sentiment via cross-validation to produce structured JSON trading executions.</div>
    </div>
</div>
""")

else:
    df = st.session_state.df

    # ── Header ──
    st.html(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 1rem;">
    <div>
        <h2 class="title-font" style="margin:0; font-size:2rem; color:#f8fafc;">{ticker}</h2>
        <div style="color:#94a3b8; font-size:0.9rem; margin-top:4px;">
            {df.index[0].strftime("%b %d")} — {df.index[-1].strftime("%b %d, %Y")} · {len(df)} bars
        </div>
    </div>
</div>
""")

    # ── KPIs ──
    price = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df) > 1 else price
    change = ((price - prev) / prev) * 100
    rsi = df["RSI"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]
    pos = "Above" if price > bb_upper else "Below" if price < bb_lower else "Inside"
    vol = df["Volume"].mean() / 1e6

    c_col = "#34d399" if change >= 0 else "#fb7185"
    c_sign = "+" if change >= 0 else ""
    r_col = "#fb7185" if rsi > 70 else "#34d399" if rsi < 30 else "#f8fafc"
    p_col = "#fb7185" if pos == "Above" else "#34d399" if pos == "Below" else "#38bdf8"

    st.html(f"""
<div class="kpi-container">
    <div class="kpi-box">
        <div class="kpi-label">Current Price</div>
        <div class="kpi-value">${price:,.2f}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">24h Change</div>
        <div class="kpi-value" style="color:{c_col};">{c_sign}{change:.2f}%</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">RSI Overlay</div>
        <div class="kpi-value" style="color:{r_col};">{rsi:.1f}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Bollinger Pos</div>
        <div class="kpi-value" style="color:{p_col};">{pos}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Volume Avg</div>
        <div class="kpi-value">{vol:.1f}M</div>
    </div>
</div>
""")

    # ── Interactive Chart ──
    st.html('<div class="glass-card" style="padding:16px; margin-bottom:24px;">')
    fig = create_plotly_chart(df, ticker=ticker, height=600)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.html('</div>')

    # ── Logic & Output Grid ──
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.html('<h3 class="title-font" style="font-size:1.4rem; margin-bottom:1rem; margin-top:0;">Information Flow</h3>')
        
        articles = st.session_state.news_articles or []
        if articles:
            html_content = ""
            for a in articles:
                pub = a.get("publisher", "NEWS")
                title = a.get("title", "")
                summary = a.get("summary", "")
                link = a.get("url", f"https://finance.yahoo.com/quote/{ticker}") # Real link or fallback
                
                html_content += f"""
<a href="{link}" target="_blank" style="text-decoration:none;">
<div class="news-card" style="transition: background 0.2s; cursor:pointer;">
    <div class="news-pub">{pub}</div>
    <div class="news-title" style="transition: color 0.2s;">{title}</div>
    <div class="news-dist">{summary[:120]}...</div>
</div>
</a>
"""
            
            full_html = f"""
            <style>
                .news-card:hover {{ background: rgba(255,255,255,0.02); }}
                .news-card:hover .news-title {{ color: #00f2fe; }}
            </style>
            <div class="glass-card" style="height: 450px; overflow-y: auto; padding:0;">
                {html_content}
            </div>
            """
            st.html(full_html)
        else:
            st.html("""
            <div class="glass-card" style="height: 450px; overflow-y: auto; padding:0; display:flex; align-items:center; justify-content:center;">
                <div style="padding: 30px; color:#64748b; text-align:center;">No news available.</div>
            </div>
            """)

    with col_r:
        st.html('<h3 class="title-font" style="font-size:1.4rem; margin-bottom:1rem; margin-top:0;">Execution Matrix</h3>')
        
        # Run Pipeline Logics
        if run_analysis:
            prog_container = st.empty()
            with prog_container.container():
                st.html('<div class="glass-card">')
                st.html('<h4 class="title-font" style="margin-top:0;">Pipeline Initialization</h4>')
                progress = st.progress(0, text="Instantiating Clean Architecture use cases...")
                
                try:
                    # Execute Use Case instead of raw pipeline functions
                    output_dir = PROJECT_ROOT / "dataset" / "live_sessions"
                    usecase = build_analyze_market_usecase(output_dir=output_dir)
                    
                    progress.progress(30, text="Executing VLM & Sentiment analysis...")
                    # The usecase fetches data, renders chart and runs the LLMs
                    decision = asyncio.run(usecase.execute(ticker=ticker, days=days))
                    
                    progress.progress(100, text="Decision rendered.")
                    st.session_state.decision = decision.model_dump()
                
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                
                st.html('</div>')
            prog_container.empty()

        # Display Final Decision
        d = st.session_state.decision
        if d:
            action = d.get("final_action", "HOLD")
            confidence = d.get("final_confidence", 0)
            vlm = d.get("vlm_signal", {})
            sentiment = d.get("sentiment", {})
            meta = d.get("meta", {})

            col_theme = "#10b981" if action == "BUY" else "#f43f5e" if action == "SELL" else "#f59e0b"
            
            st.html(f"""
<div class="decision-panel decision-{action}" style="margin-bottom:20px;">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:24px;">
        <div>
            <div style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; font-weight:600;">System Action Output</div>
            <span style="font-family:'Space Grotesk', sans-serif; font-size:3rem; font-weight:700; color:{col_theme}; line-height:1; letter-spacing:-1px;">{action}</span>
            <span class="badge-{action}" style="display:inline-block; padding:4px 10px; border-radius:6px; font-family:'JetBrains Mono'; font-weight:700; font-size:0.9rem; margin-left:12px; vertical-align:top; margin-top:8px;">
                CONF: {confidence:.0%}
            </span>
        </div>
    </div>
    
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; border-top: 1px solid rgba(255,255,255,0.05); padding-top:20px;">
        <div>
            <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Entry Target</div>
            <div class="mono-font" style="font-size:1.3rem; color:#f8fafc; font-weight:500;">${vlm.get('entry_price', 0):,.2f}</div>
        </div>
        <div>
            <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Risk Cut (SL)</div>
            <div class="mono-font" style="font-size:1.3rem; color:#fb7185; font-weight:500;">${vlm.get('stop_loss', 0):,.2f}</div>
        </div>
        <div>
            <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Target (TP)</div>
            <div class="mono-font" style="font-size:1.3rem; color:#34d399; font-weight:500;">${vlm.get('take_profit', 0):,.2f}</div>
        </div>
    </div>
</div>
""")
            
            # Sub-panels (Glassmorphism)
            st.html("""
<style>
    .stExpander { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.05) !important; background: rgba(15,17,26,0.4) !important; }
    .stExpander summary { background: transparent !important; color: #f8fafc !important; font-family: 'Space Grotesk', sans-serif; }
</style>
""")
            
            with st.expander("VISION SIGNAL (TECHNICAL ALGORITHM)"):
                st.markdown(f"""
<div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
    <div style="display:inline-block; padding:2px 8px; background:rgba(255,255,255,0.05); border-radius:4px; margin-bottom:10px;">
        <strong style="color:#f8fafc;">ACTION:</strong> <span style="font-family:'JetBrains Mono'">{vlm.get('action', '-')} ({vlm.get('confidence',0):.0%})</span>
    </div><br>
    {vlm.get('reasoning', '-')}
</div>
""", unsafe_allow_html=True)

            with st.expander("NLP SENTIMENT (MACRO)"):
                sent = sentiment.get("sentiment", "-")
                st.markdown(f"""
<div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
    <div style="display:inline-block; padding:2px 8px; background:rgba(255,255,255,0.05); border-radius:4px; margin-bottom:10px;">
        <strong style="color:#f8fafc;">SENTIMENT:</strong> <span style="font-family:'JetBrains Mono'">{sent} ({sentiment.get('intensity',0):.0%})</span>
    </div><br>
    {sentiment.get('summary', '-')}
    <br><br>
    <strong style="color:#f8fafc;">Vectors:</strong> {', '.join(sentiment.get('key_factors', []))}
</div>
""", unsafe_allow_html=True)

            with st.expander("SYSTEM METADATA JSON"):
                st.json(meta)

        else:
            st.html("""
<div class="glass-card" style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:450px; border: 1px dashed rgba(255,255,255,0.1);">
    <div style="color:rgba(255,255,255,0.1); margin-bottom:20px;">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
    </div>
    <div class="title-font" style="color:#64748b; font-size:1.1rem; text-align:center;">AWAITING COMMAND</div>
    <div style="color:#475569; font-size:0.85rem; text-align:center; max-width:250px; margin-top:8px;">Deploy models via the "RUN ANALYSIS" button in the command center.</div>
</div>
""")
