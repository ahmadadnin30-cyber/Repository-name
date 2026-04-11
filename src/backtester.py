"""
backtester.py
Backtests the Kronos trading strategy on historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Backtester:
    """
    Simple backtesting engine for the Kronos trading strategy.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.results = None

    def run(self, df: pd.DataFrame, predictor, forecast_horizon: int = 5,
            trade_amount_pct: float = 0.1) -> dict:
        """
        Run backtest on historical data.

        Args:
            df: OHLCV DataFrame
            predictor: KronosPredictor instance
            forecast_horizon: periods to forecast ahead
            trade_amount_pct: fraction of capital per trade

        Returns:
            dict with backtest results
        """
        print("Running backtest...")
        capital = self.initial_capital
        position = 0  # shares held
        trades = []
        portfolio_values = []
        min_lookback = 60  # minimum data points needed

        closes = df["Close"].values

        for i in range(min_lookback, len(df) - forecast_horizon):
            # Get data up to current point
            window = df.iloc[:i]
            ohlcv = window[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)

            # Get Kronos prediction
            result = predictor.predict(ohlcv, forecast_horizon=forecast_horizon)
            signal = result["signal"]
            current_price = closes[i]

            trade_amount = capital * trade_amount_pct

            if signal == "BUY" and position == 0 and capital >= trade_amount:
                shares = trade_amount / current_price
                position = shares
                capital -= trade_amount
                trades.append({
                    "date": df.index[i],
                    "action": "BUY",
                    "price": current_price,
                    "shares": shares,
                    "capital": capital,
                })

            elif signal == "SELL" and position > 0:
                proceeds = position * current_price
                capital += proceeds
                trades.append({
                    "date": df.index[i],
                    "action": "SELL",
                    "price": current_price,
                    "shares": position,
                    "capital": capital,
                })
                position = 0

            # Track portfolio value
            portfolio_value = capital + (position * current_price)
            portfolio_values.append({
                "date": df.index[i],
                "portfolio_value": portfolio_value,
                "price": current_price,
            })

        # Close any open position at end
        if position > 0:
            final_price = closes[-1]
            capital += position * final_price
            position = 0

        # Calculate metrics
        portfolio_df = pd.DataFrame(portfolio_values)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        final_value = capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Buy & hold comparison
        buy_hold_return = (closes[-1] - closes[min_lookback]) / closes[min_lookback] * 100

        # Sharpe ratio
        if len(portfolio_df) > 1:
            daily_returns = portfolio_df["portfolio_value"].pct_change().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        if len(portfolio_df) > 0:
            rolling_max = portfolio_df["portfolio_value"].cummax()
            drawdown = (portfolio_df["portfolio_value"] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0

        # Win rate
        if len(trades_df) > 1:
            sell_trades = trades_df[trades_df["action"] == "SELL"]
            buy_trades = trades_df[trades_df["action"] == "BUY"]
            if len(sell_trades) > 0 and len(buy_trades) > 0:
                min_len = min(len(buy_trades), len(sell_trades))
                profits = sell_trades["price"].values[:min_len] - buy_trades["price"].values[:min_len]
                win_rate = (profits > 0).mean() * 100
            else:
                win_rate = 0
        else:
            win_rate = 0

        self.results = {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "buy_hold_return_pct": buy_hold_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "total_trades": len(trades),
            "portfolio_df": portfolio_df,
            "trades_df": trades_df,
        }

        print(f"Backtest complete!")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:+.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {len(trades)}")

        return self.results


if __name__ == "__main__":
    from data_fetcher import fetch_stock_data
    from kronos_predictor import KronosPredictor

    df = fetch_stock_data("AAPL", period="1y")
    predictor = KronosPredictor(model_size="mini")
    predictor.load_model()

    backtester = Backtester(initial_capital=10000)
    results = backtester.run(df, predictor, forecast_horizon=5)
