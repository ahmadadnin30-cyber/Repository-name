"""
dashboard.py
Streamlit dashboard for Kronos AI Stock Trading System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_stock_data, format_for_kronos, get_latest_price
from kronos_predictor import KronosPredictor
from trading_bot import TradingBot
from backtester import Backtester

st.set_page_config(page_title="Kronos AI Trading", page_icon="📈", layout="wide")
st.title("📈 Kronos AI Stock Trading System")
st.markdown("*Powered by Kronos — Foundation Model for Financial Markets*")

for key in ["analysis_done", "forecast", "df", "price_info", "bt_results", "trade_result", "ticker", "forecast_days"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    period = st.selectbox("Data Period", ["3mo", "6mo", "1y", "2y"], index=1)
    model_size = st.selectbox("Kronos Model", ["mini", "small", "base"], index=0)
    forecast_days = st.slider("Forecast Days", 1, 10, 5)
    trade_amount = st.number_input("Trade Amount (USD)", value=1000, step=100)
    run_backtest = st.checkbox("Run Backtest", value=True)
    st.divider()
    st.caption("🔒 Paper Trading Mode — No real money")

@st.cache_resource
def load_predictor(size):
    p = KronosPredictor(model_size=size)
    p.load_model()
    return p

@st.cache_resource
def load_bot():
    bot = TradingBot()
    bot.connect()
    return bot

if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    st.session_state.trade_result = None
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

    st.session_state.analysis_done = True
    st.session_state.forecast = forecast
    st.session_state.df = df
    st.session_state.price_info = price_info
    st.session_state.ticker = ticker
    st.session_state.forecast_days = forecast_days

    if run_backtest:
        with st.spinner("Running backtest..."):
            backtester = Backtester(initial_capital=10000)
            bt_results = backtester.run(df, predictor, forecast_horizon=forecast_days)
            st.session_state.bt_results = bt_results

if st.session_state.analysis_done and st.session_state.forecast:
    forecast = st.session_state.forecast
    df = st.session_state.df
    price_info = st.session_state.price_info
    current_ticker = st.session_state.ticker
    current_forecast_days = st.session_state.forecast_days

    st.subheader(f"📊 {current_ticker} Analysis")
    signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${price_info['current_price']:.2f}", f"{price_info['change_pct']:+.2f}%")
    with col2:
        st.metric("AI Signal", f"{signal_color.get(forecast['signal'], '')} {forecast['signal']}")
    with col3:
        st.metric("Predicted Return", f"{forecast['predicted_return_pct']:+.2f}%")
    with col4:
        st.metric("Volatility (Ann.)", f"{forecast['volatility']:.1%}")

    st.subheader("📉 Price History & Forecast")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Historical"))
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=current_forecast_days + 1, freq="B")[1:]
    forecast_prices = [forecast["last_close"]] + forecast["forecast_prices"]
    forecast_dates = [last_date] + list(future_dates)
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode="lines+markers", name="Kronos Forecast", line=dict(color="orange", width=2, dash="dash")))
    if forecast.get("upper_bound") and forecast.get("lower_bound"):
        upper = [forecast["last_close"]] + forecast["upper_bound"]
        lower = [forecast["last_close"]] + forecast["lower_bound"]
        fig.add_trace(go.Scatter(x=forecast_dates + forecast_dates[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(255,165,0,0.15)", line=dict(color="rgba(255,255,255,0)"), name="Confidence Band"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤖 Trading Bot")
    bot = load_bot()
    account = bot.get_account_info()
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Mode:** {account.get('mode', 'simulation').upper()}\n\n**Buying Power:** ${account.get('buying_power', 0):,.2f}\n\n**Portfolio Value:** ${account.get('portfolio_value', 0):,.2f}")
    with col2:
        signal = forecast["signal"]
        btn_label = f"✅ Execute BUY for {current_ticker}" if signal == "BUY" else f"🔴 Execute SELL for {current_ticker}" if signal == "SELL" else f"⏸️ HOLD — No Trade Needed"
        if st.button(btn_label, use_container_width=True, key="trade_btn"):
            trade_result = bot.execute_signal(current_ticker, signal, trade_amount)
            st.session_state.trade_result = trade_result

    if st.session_state.trade_result:
        tr = st.session_state.trade_result
        if tr["status"] in ["EXECUTED", "SIMULATED"]:
            st.success(f"✅ Trade executed! {tr.get('message', f'{signal} {current_ticker} ${trade_amount}')}")
        elif tr["status"] == "HOLD":
            st.info("⏸️ HOLD — No trade executed")
        else:
            st.warning(f"ℹ️ {tr.get('message', tr['status'])}")

    if st.session_state.bt_results:
        bt = st.session_state.bt_results
        st.subheader("📋 Backtest Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategy Return", f"{bt['total_return_pct']:+.2f}%")
        with col2:
            st.metric("Buy & Hold Return", f"{bt['buy_hold_return_pct']:+.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{bt['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{bt['max_drawdown_pct']:.2f}%")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Win Rate", f"{bt['win_rate_pct']:.1f}%")
        with col2:
            st.metric("Total Trades", bt['total_trades'])
        if len(bt["portfolio_df"]) > 0:
            port_df = bt["portfolio_df"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=port_df["date"], y=port_df["portfolio_value"], mode="lines", name="Portfolio Value", line=dict(color="green", width=2), fill="tozeroy", fillcolor="rgba(0,200,0,0.1)"))
            fig2.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Value (USD)", height=350)
            st.plotly_chart(fig2, use_container_width=True)
        if len(bt["trades_df"]) > 0:
            st.subheader("📝 Trade History")
            st.dataframe(bt["trades_df"], use_container_width=True)

    st.success("✅ Analysis complete!")

else:
    st.info("👆 Configure settings in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### 🚀 Features
    - **Kronos AI Forecasting** — Foundation model trained on 12B+ K-line records
    - **Paper Trading** — Execute simulated trades via Alpaca
    - **Backtesting** — Test strategy on historical data
    - **Interactive Charts** — Candlestick charts with forecast overlay
    """)
