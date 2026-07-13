"""Signal processing: turn a raw RSSI stream into a motion score.

Pipeline per incoming sample:

1. EWMA baseline with a slow time constant tracks drift (AP power control,
   temperature) without absorbing fast human-motion fades.
2. The residual (sample - baseline) feeds a fixed-length ring buffer.
3. Per analysis window we compute:
   - robust sigma of residuals (MAD-scaled, immune to single-packet spikes)
   - fraction of spectral energy inside the human-motion band (0.3-2 Hz),
     which separates a walking person from wideband interference.
4. The motion score is the ratio of current sigma to the calibrated quiet-room
   sigma, weighted by band concentration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

MOTION_BAND_HZ = (0.3, 2.0)


def robust_sigma(x: np.ndarray) -> float:
    """MAD-based standard deviation estimate; robust to outlier samples."""
    if x.size < 4:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def band_energy_fraction(x: np.ndarray, rate_hz: float,
                         band: tuple = MOTION_BAND_HZ) -> float:
    """Fraction of AC spectral energy that falls inside `band`."""
    n = x.size
    if n < 16 or rate_hz <= 0:
        return 0.0
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / rate_hz)
    total = spec[1:].sum()  # skip DC
    if total <= 0:
        return 0.0
    in_band = spec[(freqs >= band[0]) & (freqs <= band[1])].sum()
    return float(in_band / total)


@dataclass
class WindowFeatures:
    sigma: float            # robust std of residuals in the window (dB)
    band_fraction: float    # share of energy in the human-motion band
    rssi: float             # latest raw sample (for display)
    n: int                  # samples in window


class FeatureExtractor:
    """Streaming feature computation over a sliding window."""

    def __init__(self, rate_hz: float, window_s: float = 4.0,
                 baseline_tau_s: float = 30.0):
        self.rate_hz = rate_hz
        self.window = deque(maxlen=max(8, int(window_s * rate_hz)))
        # EWMA alpha for a time constant of baseline_tau_s
        self._alpha = 1.0 - np.exp(-1.0 / (baseline_tau_s * rate_hz))
        self._baseline: Optional[float] = None
        self._last_raw: float = float("nan")

    def add(self, rssi: float) -> Optional[WindowFeatures]:
        self._last_raw = rssi
        if self._baseline is None:
            self._baseline = rssi
        else:
            self._baseline += self._alpha * (rssi - self._baseline)
        self.window.append(rssi - self._baseline)

        if len(self.window) < self.window.maxlen // 2:
            return None  # not enough context yet
        res = np.asarray(self.window)
        return WindowFeatures(
            sigma=robust_sigma(res),
            band_fraction=band_energy_fraction(res, self.rate_hz),
            rssi=rssi,
            n=res.size,
        )

    def reset(self) -> None:
        self.window.clear()
        self._baseline = None
