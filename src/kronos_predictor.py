"""
kronos_predictor.py
Loads the Kronos model from HuggingFace and generates price forecasts.
"""

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import torch


class KronosPredictor:
    """
    Wrapper for the Kronos financial foundation model.
    Handles loading, preprocessing, and forecasting.
    """

    MODEL_NAMES = {
        "mini": "shiyu-coder/Kronos-mini",
        "small": "shiyu-coder/Kronos-small",
        "base": "shiyu-coder/Kronos-base",
    }

    def __init__(self, model_size: str = "mini"):
        """
        Initialize the Kronos predictor.

        Args:
            model_size: 'mini', 'small', or 'base'
        """
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def load_model(self):
        """Load the Kronos model from HuggingFace."""
        try:
            from transformers import AutoModel
            model_name = self.MODEL_NAMES.get(self.model_size, self.MODEL_NAMES["mini"])
            print(f"Loading Kronos-{self.model_size} model...")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Could not load Kronos model: {e}")
            print("Falling back to statistical forecasting...")
            self.model = None

    def predict(self, ohlcv_data: np.ndarray, forecast_horizon: int = 5) -> dict:
        """
        Generate price forecast using Kronos or fallback statistical method.

        Args:
            ohlcv_data: numpy array of shape (sequence_length, 5) - OHLCV data
            forecast_horizon: number of future periods to predict

        Returns:
            dict with forecast results
        """
        if self.model is not None:
            return self._kronos_predict(ohlcv_data, forecast_horizon)
        else:
            return self._statistical_predict(ohlcv_data, forecast_horizon)

    def _kronos_predict(self, ohlcv_data: np.ndarray, forecast_horizon: int) -> dict:
        """Use Kronos model for prediction."""
        try:
            # Prepare input tensor
            input_tensor = torch.FloatTensor(ohlcv_data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor, forecast_horizon=forecast_horizon)

            forecast_prices = output.cpu().numpy().flatten()
            last_close = ohlcv_data[-1, 3]  # Last close price

            return self._build_result(last_close, forecast_prices, ohlcv_data)

        except Exception as e:
            print(f"Kronos prediction error: {e}, using fallback...")
            return self._statistical_predict(ohlcv_data, forecast_horizon)

    def _statistical_predict(self, ohlcv_data: np.ndarray, forecast_horizon: int) -> dict:
        """Fallback: simple statistical forecast using moving averages and momentum."""
        closes = ohlcv_data[:, 3]  # Close prices
        last_close = closes[-1]

        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        mean_return = np.mean(returns[-20:])  # Last 20 periods
        std_return = np.std(returns[-20:])

        # Generate forecast paths (Monte Carlo)
        n_paths = 100
        paths = []
        for _ in range(n_paths):
            path = [last_close]
            for _ in range(forecast_horizon):
                r = np.random.normal(mean_return, std_return)
                path.append(path[-1] * (1 + r))
            paths.append(path[1:])

        paths = np.array(paths)
        forecast_mean = paths.mean(axis=0)
        forecast_upper = np.percentile(paths, 75, axis=0)
        forecast_lower = np.percentile(paths, 25, axis=0)

        return self._build_result(last_close, forecast_mean, ohlcv_data,
                                  forecast_upper, forecast_lower)

    def _build_result(self, last_close, forecast_prices, ohlcv_data,
                      upper=None, lower=None) -> dict:
        """Build the result dictionary."""
        predicted_return = (forecast_prices[-1] - last_close) / last_close
        signal = "BUY" if predicted_return > 0.01 else "SELL" if predicted_return < -0.01 else "HOLD"

        # Volatility estimate
        closes = ohlcv_data[:, 3]
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        return {
            "last_close": float(last_close),
            "forecast_prices": forecast_prices.tolist(),
            "predicted_return_pct": float(predicted_return * 100),
            "signal": signal,
            "volatility": float(volatility),
            "confidence": float(max(0, 1 - volatility)),
            "upper_bound": upper.tolist() if upper is not None else None,
            "lower_bound": lower.tolist() if lower is not None else None,
        }


if __name__ == "__main__":
    from data_fetcher import fetch_stock_data, format_for_kronos

    df = fetch_stock_data("AAPL", period="3mo")
    data = format_for_kronos(df)

    predictor = KronosPredictor(model_size="mini")
    predictor.load_model()

    result = predictor.predict(data, forecast_horizon=5)
    print(f"\nForecast for AAPL:")
    print(f"Last Close: ${result['last_close']:.2f}")
    print(f"Signal: {result['signal']}")
    print(f"Predicted Return: {result['predicted_return_pct']:+.2f}%")
    print(f"Volatility: {result['volatility']:.2%}")
