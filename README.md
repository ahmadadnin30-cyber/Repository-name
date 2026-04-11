# 📈 Kronos AI Stock Trading System

AI-powered US stock trading using **Kronos** — a foundation model for financial markets from Tsinghua University.

## 🚀 Features
- **Kronos AI Forecasting** — Price prediction using a model trained on 12B+ K-line records
- **Paper Trading** — Safe simulation via Alpaca (no real money)
- **Backtesting** — Test strategy performance on historical data
- **Interactive Dashboard** — Streamlit UI with Plotly charts

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Repository-name.git
cd Repository-name
```

### 2. Create virtual environment
```bash
py -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
copy .env.example .env
```

Get free Alpaca paper trading keys at: https://alpaca.markets

### 5. Run the dashboard
```bash
streamlit run src/dashboard.py
```

## 📁 Project Structure
```
kronos-trading/
├── src/
│   ├── data_fetcher.py      # Fetch US stock data via yfinance
│   ├── kronos_predictor.py  # Kronos AI model wrapper
│   ├── trading_bot.py       # Alpaca paper trading bot
│   ├── backtester.py        # Strategy backtesting engine
│   └── dashboard.py         # Streamlit UI
├── data/                    # Local data cache (gitignored)
├── .env.example             # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚠️ Disclaimer
This is for **educational purposes only**. Not financial advice. Always do your own research before trading real money.
