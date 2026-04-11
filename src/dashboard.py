"""
dashboard.py
Streamlit dashboard for Kronos AI Stock Trading System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_stock_data, format_for_kronos, get_latest_price
from kronos_predictor import KronosPredictor
from trading_bot import TradingBot
from backtester import Backtester

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kronos AI Trading",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Kronos AI Stock Trading System")
st.markdown("*Powered by Kronos — Foundation Model for Financial Markets*")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    period = st.selectbox("Data Period", ["3mo", "6mo", "1y", "2y"], index=1)
    model_size = st.selectbox("Kronos Model", ["mini", "small", "base"], index=0)
    forecast_days = st.slider("Forecast Days", 1, 10, 5)
    trade_amount = st.number_input("Trade Amount (USD)", value=1000, step=100)

    st.divider()
    run_backtest = st.checkbox("Run Backtest", value=True)
    st.divider()
    st.caption("🔒 Paper Trading Mode — No real money")

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_predictor(size):
    p = KronosPredictor(model_size=size)
    p.load_model()
    return p

# ── Main ──────────────────────────────────────────────────────────────────────
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):

    with st.spinner(f"Fetching {ticker} data..."):
        try:
            df = fetch_stock_data(ticker, period=period)
            ohlcv = format_for_kronos(df)
            price_info = get_latest_price(ticker)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    with st.spinner("Loading Kronos model..."):
        predictor = load_predictor(model_size)

    with st.spinner("Generating forecast..."):
        forecast = predictor.predict(ohlcv, forecast_horizon=forecast_days)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.subheader(f"📊 {ticker} Analysis")
    col1, col2, col3, col4 = st.columns(4)

    signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    with col1:
        st.metric("Current Price", f"${price_info['current_price']:.2f}",
                  f"{price_info['change_pct']:+.2f}%")
    with col2:
        st.metric("AI Signal", f"{signal_color.get(forecast['signal'], '')} {forecast['signal']}")
    with col3:
        st.metric("Predicted Return", f"{forecast['predicted_return_pct']:+.2f}%")
    with col4:
        st.metric("Volatility (Ann.)", f"{forecast['volatility']:.1%}")

    # ── Price chart with forecast ─────────────────────────────────────────────
    st.subheader("📉 Price History & Forecast")

    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Historical",
    ))

    # Forecast line
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq="B")[1:]
    forecast_prices = [forecast["last_close"]] + forecast["forecast_prices"]
    forecast_dates = [last_date] + list(future_dates)

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode="lines+markers",
        name="Kronos Forecast",
        line=dict(color="orange", width=2, dash="dash"),
        marker=dict(size=6),
    ))

    # Confidence bands
    if forecast.get("upper_bound") and forecast.get("lower_bound"):
        upper = [forecast["last_close"]] + forecast["upper_bound"]
        lower = [forecast["last_close"]] + forecast["lower_bound"]
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band",
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Trading bot ───────────────────────────────────────────────────────────
    st.subheader("🤖 Trading Bot")
    bot = TradingBot()
    bot.connect()
    account = bot.get_account_info()

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Mode:** {account.get('mode', 'simulation').upper()}\n\n"
                f"**Buying Power:** ${account.get('buying_power', 0):,.2f}\n\n"
                f"**Portfolio Value:** ${account.get('portfolio_value', 0):,.2f}")
    with col2:
        if st.button(f"Execute {forecast['signal']} Signal", use_container_width=True):
            trade_result = bot.execute_signal(ticker, forecast["signal"], trade_amount)
            if trade_result["status"] in ["EXECUTED", "SIMULATED"]:
                st.success(f"✅ {trade_result['message'] if 'message' in trade_result else 'Trade executed!'}")
            else:
                st.info(f"ℹ️ {trade_result.get('message', trade_result['status'])}")

    # ── Backtest ──────────────────────────────────────────────────────────────
    if run_backtest:
        st.subheader("📋 Backtest Results")
        with st.spinner("Running backtest..."):
            backtester = Backtester(initial_capital=10000)
            bt_results = backtester.run(df, predictor, forecast_horizon=forecast_days)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategy Return", f"{bt_results['total_return_pct']:+.2f}%")
        with col2:
            st.metric("Buy & Hold Return", f"{bt_results['buy_hold_return_pct']:+.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{bt_results['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{bt_results['max_drawdown_pct']:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Win Rate", f"{bt_results['win_rate_pct']:.1f}%")
        with col2:
            st.metric("Total Trades", bt_results['total_trades'])

        # Portfolio value chart
        if len(bt_results["portfolio_df"]) > 0:
            port_df = bt_results["portfolio_df"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=port_df["date"],
                y=port_df["portfolio_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="green", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,200,0,0.1)",
            ))
            fig2.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Value (USD)",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Trade history
        if len(bt_results["trades_df"]) > 0:
            st.subheader("📝 Trade History")
            st.dataframe(bt_results["trades_df"], use_container_width=True)

    st.success("✅ Analysis complete!")

else:
    st.info("👆 Configure settings in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### 🚀 Features
    - **Kronos AI Forecasting** — Foundation model trained on 12B+ K-line records
    - **Paper Trading** — Execute simulated trades via Alpaca
    - **Backtesting** — Test strategy on historical data
    - **Interactive Charts** — Candlestick charts with forecast overlay

    ### 📦 Setup
    1. Add your Alpaca API keys to `.env` file
    2. Select a stock ticker and settings
    3. Click Run Analysis!
    """)
