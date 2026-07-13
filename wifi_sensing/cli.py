"""Command line interface.

  python -m wifi_sensing calibrate [--seconds 30]     # record quiet room
  python -m wifi_sensing run [--threshold 2.5]        # live detection
  python -m wifi_sensing record --out session.csv     # log raw RSSI
  python -m wifi_sensing replay session.csv           # re-run detection offline
  python -m wifi_sensing run --simulate               # demo without WiFi
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Optional

import numpy as np

from . import backends
from .detector import DEFAULT_CAL_PATH, Calibration, MotionDetector
from .processing import FeatureExtractor, robust_sigma


def _make_backend(args, quiet_sim: bool = False) -> backends.Backend:
    if getattr(args, "simulate", False):
        # Calibration simulates an empty room (no motion bursts).
        return backends.Simulate(rate_hz=args.rate,
                                 motion_duty=0.0 if quiet_sim else 0.4)
    if getattr(args, "replay_file", None):
        return backends.Replay(args.replay_file)
    return backends.autodetect(getattr(args, "interface", None))


def _sample_loop(backend: backends.Backend, rate_hz: float):
    """Yield (timestamp, rssi) at up to rate_hz, sleeping only as needed."""
    period = 1.0 / min(rate_hz, backend.max_rate_hz)
    offline = isinstance(backend, (backends.Replay, backends.Simulate))
    next_t = time.monotonic()
    while True:
        v = backend.read()
        if v is None:
            if isinstance(backend, backends.Replay):
                return  # file exhausted
            time.sleep(1.0)  # link dropped; wait and retry
            continue
        yield time.time(), v
        if not offline:
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()


def cmd_calibrate(args) -> int:
    backend = _make_backend(args, quiet_sim=True)
    rate = min(args.rate, backend.max_rate_hz)
    n_target = int(args.seconds * rate)
    print(f"Backend: {backend.name} @ {rate:.1f} Hz")
    print(f"Calibrating for {args.seconds:.0f}s — keep the room EMPTY and still...")

    extractor = FeatureExtractor(rate_hz=rate)
    residual_sigmas = []
    samples = 0
    for _, rssi in _sample_loop(backend, rate):
        feat = extractor.add(rssi)
        samples += 1
        if feat is not None:
            residual_sigmas.append(feat.sigma)
        if samples % max(1, int(rate)) == 0:
            print(f"\r  {samples}/{n_target} samples", end="", flush=True)
        if samples >= n_target:
            break
    print()
    backend.close()

    if len(residual_sigmas) < 5:
        print("Not enough samples collected — try a longer calibration.", file=sys.stderr)
        return 1

    # Use a high percentile of quiet-room sigma so normal idle wobble
    # doesn't cross the threshold.
    idle_sigma = float(np.percentile(residual_sigmas, 90))
    cal = Calibration(idle_sigma=idle_sigma, rate_hz=rate, backend=backend.name)
    cal.save(args.cal_file)
    print(f"Idle sigma: {idle_sigma:.3f} dB  ->  saved to {args.cal_file}")
    print("Now run:  python -m wifi_sensing run")
    return 0


def _load_or_default_cal(args, backend, rate) -> Calibration:
    try:
        cal = Calibration.load(args.cal_file)
        if abs(cal.rate_hz - rate) > 0.5:
            print(f"note: calibration was taken at {cal.rate_hz:.1f} Hz, "
                  f"running at {rate:.1f} Hz", file=sys.stderr)
        return cal
    except (OSError, KeyError, ValueError):
        print("No calibration found — using a conservative default. "
              "Run `calibrate` for best accuracy.", file=sys.stderr)
        return Calibration(idle_sigma=0.6, rate_hz=rate, backend=backend.name)


def cmd_run(args) -> int:
    backend = _make_backend(args)
    rate = min(args.rate, backend.max_rate_hz)
    cal = _load_or_default_cal(args, backend, rate)

    extractor = FeatureExtractor(rate_hz=rate)
    detector = MotionDetector(
        cal, threshold=args.threshold,
        presence_hold_s=args.presence_hold,
    )
    print(f"Backend: {backend.name} @ {rate:.1f} Hz | idle sigma {cal.sigma:.3f} dB "
          f"| threshold {args.threshold}x")
    print("Watching for motion — Ctrl+C to stop.\n")

    log = None
    if args.out:
        log = open(args.out, "w", newline="")
        writer = csv.writer(log)
        writer.writerow(["timestamp", "rssi", "score", "motion", "occupied"])

    was_motion = False
    try:
        for ts, rssi in _sample_loop(backend, rate):
            feat = extractor.add(rssi)
            if feat is None:
                continue
            state = detector.update(feat)

            if state.motion and not was_motion:
                print(f"\n[{time.strftime('%H:%M:%S')}] MOTION detected")
            elif was_motion and not state.motion:
                print(f"\n[{time.strftime('%H:%M:%S')}] motion ended")
            was_motion = state.motion

            bar_len = min(30, int(state.score * 6))
            label = "MOTION " if state.motion else ("present" if state.occupied else "idle   ")
            print(f"\r{label} | rssi {state.rssi:6.1f} dBm | score {state.score:5.2f} "
                  f"|{'#' * bar_len}{' ' * (30 - bar_len)}|", end="", flush=True)

            if log:
                writer.writerow([f"{ts:.3f}", f"{rssi:.2f}", f"{state.score:.3f}",
                                 int(state.motion), int(state.occupied)])
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if log:
            log.close()
        backend.close()
    return 0


def cmd_record(args) -> int:
    backend = _make_backend(args)
    rate = min(args.rate, backend.max_rate_hz)
    print(f"Backend: {backend.name} @ {rate:.1f} Hz -> {args.out}  (Ctrl+C to stop)")
    n = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rssi"])
        try:
            for ts, rssi in _sample_loop(backend, rate):
                writer.writerow([f"{ts:.3f}", f"{rssi:.2f}"])
                n += 1
                if n % max(1, int(rate)) == 0:
                    print(f"\r  {n} samples", end="", flush=True)
                if args.seconds and n >= args.seconds * rate:
                    break
        except KeyboardInterrupt:
            pass
    print(f"\nwrote {n} samples to {args.out}")
    backend.close()
    return 0


def cmd_replay(args) -> int:
    args.replay_file = args.file
    args.simulate = False
    backend = _make_backend(args)
    rate = args.rate
    cal = _load_or_default_cal(args, backend, rate)
    extractor = FeatureExtractor(rate_hz=rate)
    detector = MotionDetector(cal, threshold=args.threshold)

    events = []
    was_motion = False
    t = 0.0
    for _, rssi in _sample_loop(backend, rate):
        t += 1.0 / rate
        feat = extractor.add(rssi)
        if feat is None:
            continue
        state = detector.update(feat, now=t)
        if state.motion != was_motion:
            events.append((t, "MOTION" if state.motion else "quiet"))
            was_motion = state.motion
    if events:
        print("Detected transitions:")
        for t, kind in events:
            print(f"  t={t:7.1f}s  {kind}")
    else:
        print("No motion detected in recording.")
    return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="wifi_sensing",
                                description="Detect human motion/presence from laptop WiFi RSSI.")
    p.add_argument("--rate", type=float, default=10.0, help="target sample rate, Hz")
    p.add_argument("--interface", help="wireless interface (default: auto)")
    p.add_argument("--cal-file", default=DEFAULT_CAL_PATH, help="calibration file path")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="measure quiet-room baseline (room must be empty)")
    pc.add_argument("--seconds", type=float, default=30.0)
    pc.add_argument("--simulate", action="store_true", help="use synthetic signal")
    pc.set_defaults(fn=cmd_calibrate)

    pr = sub.add_parser("run", help="live motion/presence detection")
    pr.add_argument("--threshold", type=float, default=2.5,
                    help="motion score threshold (x idle sigma)")
    pr.add_argument("--presence-hold", type=float, default=60.0,
                    help="seconds 'occupied' persists after motion stops")
    pr.add_argument("--out", help="also log CSV of samples/decisions")
    pr.add_argument("--simulate", action="store_true", help="use synthetic signal")
    pr.set_defaults(fn=cmd_run)

    pg = sub.add_parser("record", help="log raw RSSI to CSV")
    pg.add_argument("--out", required=True)
    pg.add_argument("--seconds", type=float, help="stop after N seconds")
    pg.add_argument("--simulate", action="store_true", help="use synthetic signal")
    pg.set_defaults(fn=cmd_record)

    pp = sub.add_parser("replay", help="run detection over a recorded CSV")
    pp.add_argument("file")
    pp.add_argument("--threshold", type=float, default=2.5)
    pp.set_defaults(fn=cmd_replay)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
