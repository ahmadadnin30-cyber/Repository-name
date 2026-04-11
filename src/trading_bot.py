"""
trading_bot.py
Executes paper trades on Alpaca based on Kronos signals.
"""

import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class TradingBot:
    """
    Paper trading bot using Alpaca API.
    Executes trades based on Kronos AI signals.
    """

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.api = None
        self.trade_log = []

    def connect(self):
        """Connect to Alpaca API."""
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                self.base_url,
                api_version="v2"
            )
            account = self.api.get_account()
            print(f"Connected to Alpaca!")
            print(f"Account Status: {account.status}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            return True
        except Exception as e:
            print(f"Could not connect to Alpaca: {e}")
            print("Running in simulation mode...")
            return False

    def get_account_info(self) -> dict:
        """Get current account information."""
        if self.api is None:
            return {"mode": "simulation", "buying_power": 100000, "portfolio_value": 100000}

        try:
            account = self.api.get_account()
            return {
                "mode": "paper",
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "equity": float(account.equity),
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}

    def get_position(self, ticker: str) -> dict:
        """Get current position for a ticker."""
        if self.api is None:
            return {"qty": 0, "avg_entry": 0, "current_value": 0}

        try:
            position = self.api.get_position(ticker)
            return {
                "qty": float(position.qty),
                "avg_entry": float(position.avg_entry_price),
                "current_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
            }
        except:
            return {"qty": 0, "avg_entry": 0, "current_value": 0}

    def execute_signal(self, ticker: str, signal: str, amount_usd: float = 1000) -> dict:
        """
        Execute a trade based on Kronos signal.

        Args:
            ticker: Stock symbol
            signal: 'BUY', 'SELL', or 'HOLD'
            amount_usd: Dollar amount to trade

        Returns:
            Trade result dict
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if signal == "HOLD":
            result = {"status": "HOLD", "ticker": ticker, "message": "No trade executed", "timestamp": timestamp}
            self.trade_log.append(result)
            return result

        if self.api is None:
            # Simulation mode
            result = {
                "status": "SIMULATED",
                "ticker": ticker,
                "signal": signal,
                "amount_usd": amount_usd,
                "message": f"Simulated {signal} ${amount_usd} of {ticker}",
                "timestamp": timestamp,
            }
            self.trade_log.append(result)
            print(f"[SIMULATION] {signal} ${amount_usd} of {ticker}")
            return result

        try:
            if signal == "BUY":
                order = self.api.submit_order(
                    symbol=ticker,
                    notional=amount_usd,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
            elif signal == "SELL":
                position = self.get_position(ticker)
                if position["qty"] > 0:
                    order = self.api.submit_order(
                        symbol=ticker,
                        qty=position["qty"],
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )
                else:
                    return {"status": "SKIP", "message": "No position to sell", "timestamp": timestamp}

            result = {
                "status": "EXECUTED",
                "ticker": ticker,
                "signal": signal,
                "order_id": order.id,
                "amount_usd": amount_usd,
                "timestamp": timestamp,
            }
            self.trade_log.append(result)
            print(f"Order executed: {signal} {ticker} - Order ID: {order.id}")
            return result

        except Exception as e:
            result = {"status": "ERROR", "ticker": ticker, "error": str(e), "timestamp": timestamp}
            self.trade_log.append(result)
            print(f"Trade error: {e}")
            return result

    def get_trade_history(self) -> list:
        """Return trade log."""
        return self.trade_log


if __name__ == "__main__":
    bot = TradingBot()
    bot.connect()
    account = bot.get_account_info()
    print(f"\nAccount: {account}")
