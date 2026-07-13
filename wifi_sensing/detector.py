"""Motion/presence detection state machine on top of the feature stream.

States:
- IDLE:    channel statistics match the calibrated quiet room.
- MOTION:  sustained excess fluctuation → someone is moving right now.
- Presence (`occupied`) latches for `presence_hold_s` after the last motion,
  covering a person who sat down and stopped moving.

Hysteresis (N windows to enter, M to leave) suppresses flicker from single
noisy windows.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

from .processing import WindowFeatures

DEFAULT_CAL_PATH = os.path.join(os.path.expanduser("~"), ".wifi_sensing_cal.json")


@dataclass
class Calibration:
    idle_sigma: float           # quiet-room robust sigma (dB)
    rate_hz: float
    backend: str = "?"
    created_at: float = field(default_factory=time.time)

    # Floor so a perfectly still channel doesn't make the ratio explode.
    MIN_SIGMA = 0.15

    def save(self, path: str = DEFAULT_CAL_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_CAL_PATH) -> "Calibration":
        with open(path) as f:
            d = json.load(f)
        return cls(idle_sigma=d["idle_sigma"], rate_hz=d["rate_hz"],
                   backend=d.get("backend", "?"), created_at=d.get("created_at", 0))

    @property
    def sigma(self) -> float:
        return max(self.idle_sigma, self.MIN_SIGMA)


@dataclass
class DetectorState:
    motion: bool
    occupied: bool
    score: float
    rssi: float
    last_motion_at: Optional[float]


class MotionDetector:
    def __init__(self, calibration: Calibration,
                 threshold: float = 2.5,
                 on_windows: int = 2,
                 off_windows: int = 6,
                 presence_hold_s: float = 60.0,
                 band_boost: float = 0.5):
        """
        threshold:  score above which a window counts as active.
        on_windows: consecutive active windows required to enter MOTION.
        off_windows: consecutive quiet windows required to leave MOTION.
        presence_hold_s: how long `occupied` stays true after motion stops.
        band_boost: how much energy concentration in the human-motion band
                    amplifies the score (0 = ignore spectrum).
        """
        self.cal = calibration
        self.threshold = threshold
        self.on_windows = on_windows
        self.off_windows = off_windows
        self.presence_hold_s = presence_hold_s
        self.band_boost = band_boost

        self._above = 0
        self._below = 0
        self._motion = False
        self._last_motion_at: Optional[float] = None

    def score(self, feat: WindowFeatures) -> float:
        ratio = feat.sigma / self.cal.sigma
        # A walking person concentrates energy at 0.3-2 Hz; reward that.
        return ratio * (1.0 + self.band_boost * feat.band_fraction)

    def update(self, feat: WindowFeatures, now: Optional[float] = None) -> DetectorState:
        now = time.time() if now is None else now
        s = self.score(feat)

        if s >= self.threshold:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            self._above = 0

        if not self._motion and self._above >= self.on_windows:
            self._motion = True
        elif self._motion and self._below >= self.off_windows:
            self._motion = False

        if self._motion:
            self._last_motion_at = now

        occupied = (
            self._last_motion_at is not None
            and (now - self._last_motion_at) <= self.presence_hold_s
        )
        return DetectorState(
            motion=self._motion,
            occupied=occupied,
            score=s,
            rssi=feat.rssi,
            last_motion_at=self._last_motion_at,
        )
