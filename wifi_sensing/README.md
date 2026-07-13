# WiFi Sensing — laptop-only human motion & presence detection

Detects when a person moves through a room using **nothing but your laptop's
WiFi connection**. No camera, no ESP32, no special drivers, no root.

## How it works

A human body is mostly water and absorbs/reflects 2.4/5 GHz radio waves. When
someone moves between (or near) your laptop and the router, the multipath
pattern of the WiFi channel shifts and the received signal strength (RSSI)
fluctuates far more than in an empty room.

This tool samples RSSI ~10x/second using the cheapest source the OS offers,
then runs a small signal-processing pipeline:

```
RSSI ──► EWMA baseline (removes slow drift)
      ──► sliding 4 s window of residuals
      ──► robust sigma (MAD)  +  0.3–2 Hz band energy (human-motion band)
      ──► score = sigma / calibrated idle sigma
      ──► hysteresis state machine ──► MOTION / present / idle
```

Why this is the *efficient* design:

| Platform | Sampling source | Cost per sample |
|----------|-----------------|-----------------|
| Linux    | `/proc/net/wireless` (file read) | ~microseconds, no subprocess |
| Windows  | `wlanapi.dll` via ctypes | native call, no subprocess |
| macOS    | `airport -I` | one short subprocess (capped 4 Hz) |

The whole thing is numpy + stdlib. CPU usage is negligible (<1% of one core).

> **Why not CSI?** Projects like esp-csi/ESPectre use Channel State
> Information, which is richer than RSSI — but extracting CSI requires
> specific radios (ESP32, Intel 5300, Atheros, or nexmon-patched Broadcom).
> A generic laptop NIC does not expose it. RSSI variance sensing is the best
> technique that works on *any* laptop, out of the box.

## Quick start

```bash
pip install numpy

# 1. Calibrate with the room EMPTY and the laptop where it will live (30 s)
python -m wifi_sensing calibrate

# 2. Detect
python -m wifi_sensing run
```

Output:

```
Backend: proc_net_wireless @ 10.0 Hz | idle sigma 0.412 dB | threshold 2.5x
idle    | rssi  -54.0 dBm | score  0.93 |#####                         |
[14:32:07] MOTION detected
MOTION  | rssi  -51.2 dBm | score  4.81 |############################  |
```

No WiFi around, or want a demo? `python -m wifi_sensing run --simulate`.

### All commands

```bash
python -m wifi_sensing calibrate [--seconds 30]   # quiet-room baseline
python -m wifi_sensing run [--threshold 2.5] [--presence-hold 60] [--out log.csv]
python -m wifi_sensing record --out session.csv [--seconds 120]
python -m wifi_sensing replay session.csv         # offline detection on a recording
```

Global flags: `--rate HZ` (default 10), `--interface wlan0`, `--cal-file PATH`.

## Tuning

- **False positives** (fires with nobody there): raise `--threshold` (try 3.5),
  or re-calibrate — a microwave, fan, or a neighbor's traffic can raise the
  idle baseline.
- **Misses** (person walks by, nothing fires): lower `--threshold` (try 1.8),
  and position the laptop so the person crosses the laptop↔router path —
  sensitivity is highest on that line.
- **`occupied` releases too early** for a person sitting still: increase
  `--presence-hold` (seconds). RSSI sensing sees *motion*, not static bodies;
  presence is inferred by latching recent motion.

## Placement tips

Best geometry: laptop and router on opposite sides of the area you care
about, so a person walking through cuts the direct path. Detection range is
roughly a room; through-wall detection works but is weaker.

## Limitations (honest ones)

- Detects **moving** people. A perfectly still person is invisible after
  `presence-hold` expires — that's physics of RSSI, not a bug.
- Cannot count people, localize them, or estimate pose — that requires CSI
  hardware (see [espressif/esp-csi](https://github.com/espressif/esp-csi) if
  you later add a $5 ESP32).
- Pets, fans, and big RF interference changes can trigger it; calibration and
  the 0.3–2 Hz band weighting suppress most of this but not all.

## Tests

```bash
python -m unittest discover -s wifi_sensing/tests -v
```

Runs the full pipeline over synthetic idle/motion signals and asserts no
false positives on a quiet channel and reliable detection during motion.

## Ethics

Only sense spaces you own or have permission to monitor. WiFi sensing sees
through walls to a degree — don't point it at your neighbors.
