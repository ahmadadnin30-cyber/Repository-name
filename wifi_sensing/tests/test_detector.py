"""End-to-end tests on synthetic RSSI: the detector must fire during motion
segments and stay quiet on an idle channel."""

import unittest

from wifi_sensing.backends import Simulate
from wifi_sensing.detector import Calibration, MotionDetector
from wifi_sensing.processing import FeatureExtractor, band_energy_fraction, robust_sigma

import numpy as np

RATE = 10.0


def run_pipeline(backend: Simulate, seconds: float, detector: MotionDetector,
                 extractor: FeatureExtractor):
    """Feed `seconds` of simulated samples; return list of (t, state, truth)."""
    out = []
    t = 0.0
    for _ in range(int(seconds * RATE)):
        t += 1.0 / RATE
        truth = backend.motion_active()
        feat = extractor.add(backend.read())
        if feat is None:
            continue
        out.append((t, detector.update(feat, now=t), truth))
    return out


class TestProcessing(unittest.TestCase):
    def test_robust_sigma_matches_std_for_gaussian(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 2.0, 5000)
        self.assertAlmostEqual(robust_sigma(x), 2.0, delta=0.15)

    def test_robust_sigma_ignores_outliers(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1.0, 1000)
        x[::100] += 50  # gross spikes
        self.assertLess(robust_sigma(x), 1.3)

    def test_band_energy_prefers_motion_band(self):
        rate = 10.0
        t = np.arange(0, 30, 1 / rate)
        in_band = np.sin(2 * np.pi * 0.9 * t)   # 0.9 Hz — walking-body fade
        out_band = np.sin(2 * np.pi * 4.0 * t)  # 4 Hz — outside band
        self.assertGreater(band_energy_fraction(in_band, rate), 0.8)
        self.assertLess(band_energy_fraction(out_band, rate), 0.2)


class TestDetector(unittest.TestCase):
    def _detector(self, idle_sigma=0.45):
        cal = Calibration(idle_sigma=idle_sigma, rate_hz=RATE, backend="simulate")
        return MotionDetector(cal, threshold=2.5, presence_hold_s=15.0)

    def test_idle_channel_stays_quiet(self):
        backend = Simulate(rate_hz=RATE, motion_duty=0.0, seed=42)
        results = run_pipeline(backend, 120, self._detector(), FeatureExtractor(RATE))
        motion_windows = sum(1 for _, s, _ in results if s.motion)
        self.assertEqual(motion_windows, 0,
                         "false positives on a quiet channel")

    def test_motion_bursts_are_detected(self):
        backend = Simulate(rate_hz=RATE, motion_period_s=30.0, motion_duty=0.5, seed=7)
        results = run_pipeline(backend, 120, self._detector(), FeatureExtractor(RATE))
        during = [s.motion for _, s, truth in results if truth]
        outside = [s.motion for _, s, truth in results if not truth]
        self.assertGreater(np.mean(during), 0.5,
                           "missed most of the motion segments")
        # Hysteresis + presence-lag means some spillover is OK, but the
        # detector must mostly release between bursts.
        self.assertLess(np.mean(outside), 0.5)

    def test_presence_latches_after_motion(self):
        backend = Simulate(rate_hz=RATE, motion_period_s=60.0, motion_duty=0.25, seed=3)
        results = run_pipeline(backend, 60, self._detector(), FeatureExtractor(RATE))
        # Just after the 15s motion burst ends, occupied must still be true.
        post = [s for t, s, truth in results if 16 <= t <= 25]
        self.assertTrue(all(s.occupied for s in post),
                        "presence should hold after motion stops")

    def test_score_scales_with_calibration(self):
        cal = Calibration(idle_sigma=0.0, rate_hz=RATE)  # clamps to MIN_SIGMA
        self.assertEqual(cal.sigma, Calibration.MIN_SIGMA)


class TestTermuxParse(unittest.TestCase):
    def test_valid_output(self):
        from wifi_sensing.backends import TermuxApi
        out = ('{"bssid":"aa:bb","frequency_mhz":2437,"ip":"192.168.1.5",'
               '"link_speed_mbps":72,"rssi":-55,"ssid":"Home",'
               '"supplicant_state":"COMPLETED"}')
        self.assertEqual(TermuxApi.parse(out), -55.0)

    def test_invalid_rssi_sentinel(self):
        from wifi_sensing.backends import TermuxApi
        self.assertIsNone(TermuxApi.parse('{"rssi": -127}'))

    def test_missing_rssi(self):
        from wifi_sensing.backends import TermuxApi
        self.assertIsNone(TermuxApi.parse('{"ssid": "Home"}'))

    def test_garbage_output(self):
        from wifi_sensing.backends import TermuxApi
        self.assertIsNone(TermuxApi.parse("error: API not available"))
        self.assertIsNone(TermuxApi.parse(""))


class TestLowRatePipeline(unittest.TestCase):
    def test_detection_works_at_termux_rate(self):
        """At 1 Hz (Android RSSI refresh) motion must still be caught."""
        from wifi_sensing.cli import _extractor
        rate = 1.0
        backend = Simulate(rate_hz=rate, motion_period_s=120.0,
                           motion_duty=0.5, seed=11)
        cal = Calibration(idle_sigma=0.45, rate_hz=rate, backend="termux")
        det = MotionDetector(cal, threshold=2.5, presence_hold_s=30.0)
        ext = _extractor(rate)
        self.assertGreaterEqual(ext.window.maxlen, 16)

        fired_during_motion = False
        t = 0.0
        for _ in range(240):
            t += 1.0 / rate
            truth = backend.motion_active()
            feat = ext.add(backend.read())
            if feat is None:
                continue
            if det.update(feat, now=t).motion and truth:
                fired_during_motion = True
        self.assertTrue(fired_during_motion,
                        "1 Hz pipeline never detected the motion segment")


class TestReplayRoundTrip(unittest.TestCase):
    def test_record_then_replay(self):
        import csv, os, tempfile
        backend = Simulate(rate_hz=RATE, motion_period_s=20.0, motion_duty=0.4, seed=5)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="") as f:
            path = f.name
            w = csv.writer(f)
            w.writerow(["timestamp", "rssi"])
            for i in range(600):
                w.writerow([i / RATE, f"{backend.read():.2f}"])
        try:
            from wifi_sensing.backends import Replay
            replay = Replay(path)
            det = MotionDetector(Calibration(idle_sigma=0.45, rate_hz=RATE),
                                 threshold=2.5)
            ext = FeatureExtractor(RATE)
            fired = False
            t = 0.0
            while not replay.exhausted:
                t += 1 / RATE
                feat = ext.add(replay.read())
                if feat and det.update(feat, now=t).motion:
                    fired = True
            self.assertTrue(fired, "replay of motion recording never fired")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
