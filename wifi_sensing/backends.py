"""RSSI sample sources.

Each backend exposes ``read() -> float | None`` returning the current RSSI in
dBm (or a dBm-like scale) for the connected WiFi link, and ``max_rate_hz``
describing how fast it can reasonably be polled.

Backends, most efficient first per platform:

- Linux:   ProcNetWireless (reads /proc/net/wireless — no subprocess)
           IwLink          (subprocess fallback: ``iw dev <if> link``)
- Windows: WlanApi         (ctypes into wlanapi.dll — no subprocess)
           Netsh           (subprocess fallback: ``netsh wlan show interfaces``)
- macOS:   Airport         (subprocess: the airport private binary)
- Any:     Replay          (CSV file, for offline analysis/testing)
           Simulate        (synthetic idle/motion signal, for demos/tests)
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Optional


class Backend:
    name = "base"
    max_rate_hz = 10.0

    def read(self) -> Optional[float]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------------

class ProcNetWireless(Backend):
    """Signal level from /proc/net/wireless. Cheapest possible source on Linux."""

    name = "proc_net_wireless"
    max_rate_hz = 20.0
    PATH = "/proc/net/wireless"

    def __init__(self, interface: Optional[str] = None):
        self.interface = interface
        if self.read() is None:
            raise RuntimeError("no wireless interface with signal data in /proc/net/wireless")

    def read(self) -> Optional[float]:
        try:
            with open(self.PATH) as f:
                lines = f.readlines()[2:]
        except OSError:
            return None
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            iface = parts[0].rstrip(":")
            if self.interface and iface != self.interface:
                continue
            try:
                return float(parts[3].rstrip("."))
            except ValueError:
                continue
        return None


class IwLink(Backend):
    """Fallback: parse ``iw dev <interface> link``."""

    name = "iw"
    max_rate_hz = 5.0
    _SIGNAL_RE = re.compile(r"signal:\s*(-?\d+)\s*dBm")

    def __init__(self, interface: Optional[str] = None):
        self.interface = interface or self._find_interface()
        if self.interface is None or self.read() is None:
            raise RuntimeError("iw found no connected wireless interface")

    @staticmethod
    def _find_interface() -> Optional[str]:
        try:
            out = subprocess.run(
                ["iw", "dev"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        m = re.search(r"Interface\s+(\S+)", out)
        return m.group(1) if m else None

    def read(self) -> Optional[float]:
        try:
            out = subprocess.run(
                ["iw", "dev", self.interface, "link"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        m = self._SIGNAL_RE.search(out)
        return float(m.group(1)) if m else None


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

class WlanApi(Backend):
    """Native Windows WLAN API via ctypes. No subprocess per sample.

    Reads wlanSignalQuality (0-100) from the current connection and maps it
    to a dBm-like value with the standard quality/2 - 100 conversion.
    """

    name = "wlanapi"
    max_rate_hz = 20.0

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("wlanapi backend is Windows-only")
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._wlanapi = ctypes.windll.wlanapi
        self._kernel32 = ctypes.windll.kernel32

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class WLAN_INTERFACE_INFO(ctypes.Structure):
            _fields_ = [
                ("InterfaceGuid", GUID),
                ("strInterfaceDescription", wintypes.WCHAR * 256),
                ("isState", ctypes.c_uint),
            ]

        class WLAN_INTERFACE_INFO_LIST(ctypes.Structure):
            _fields_ = [
                ("dwNumberOfItems", wintypes.DWORD),
                ("dwIndex", wintypes.DWORD),
                ("InterfaceInfo", WLAN_INTERFACE_INFO * 1),
            ]

        self._GUID = GUID
        self._INFO_LIST = WLAN_INTERFACE_INFO_LIST

        handle = wintypes.HANDLE()
        negotiated = wintypes.DWORD()
        if self._wlanapi.WlanOpenHandle(2, None, ctypes.byref(negotiated), ctypes.byref(handle)):
            raise RuntimeError("WlanOpenHandle failed")
        self._handle = handle

        ifaces = ctypes.POINTER(WLAN_INTERFACE_INFO_LIST)()
        if self._wlanapi.WlanEnumInterfaces(handle, None, ctypes.byref(ifaces)):
            raise RuntimeError("WlanEnumInterfaces failed")
        try:
            if ifaces.contents.dwNumberOfItems == 0:
                raise RuntimeError("no WLAN interfaces")
            self._guid = GUID()
            ctypes.memmove(
                ctypes.byref(self._guid),
                ctypes.byref(ifaces.contents.InterfaceInfo[0].InterfaceGuid),
                ctypes.sizeof(GUID),
            )
        finally:
            self._wlanapi.WlanFreeMemory(ifaces)

        if self.read() is None:
            raise RuntimeError("WLAN interface is not connected")

    def read(self) -> Optional[float]:
        ctypes = self._ct
        from ctypes import wintypes

        # wlan_intf_opcode_current_connection = 7; wlanSignalQuality is the
        # DWORD at offset 4 within the WLAN_ASSOCIATION_ATTRIBUTES that starts
        # at offset 532 of WLAN_CONNECTION_ATTRIBUTES:
        #   isState(4) + wlanConnectionMode(4) + strProfileName(2*256) = 520
        #   then dot11Ssid(4 + 32 = 36) + dot11BssType(4) + dot11Bssid(6+pad 2)
        #   -> quality at 520 + 36 + 4 + 8 + 4(phytype... ) ; rather than hand
        # computing fragile offsets we define the real structures.
        class DOT11_SSID(ctypes.Structure):
            _fields_ = [("uSSIDLength", wintypes.ULONG), ("ucSSID", ctypes.c_char * 32)]

        class WLAN_ASSOCIATION_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("dot11Ssid", DOT11_SSID),
                ("dot11BssType", ctypes.c_uint),
                ("dot11Bssid", ctypes.c_ubyte * 6),
                ("dot11PhyType", ctypes.c_uint),
                ("uDot11PhyIndex", wintypes.ULONG),
                ("wlanSignalQuality", wintypes.ULONG),
                ("ulRxRate", wintypes.ULONG),
                ("ulTxRate", wintypes.ULONG),
            ]

        class WLAN_SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("bSecurityEnabled", wintypes.BOOL),
                ("bOneXEnabled", wintypes.BOOL),
                ("dot11AuthAlgorithm", ctypes.c_uint),
                ("dot11CipherAlgorithm", ctypes.c_uint),
            ]

        class WLAN_CONNECTION_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("isState", ctypes.c_uint),
                ("wlanConnectionMode", ctypes.c_uint),
                ("strProfileName", wintypes.WCHAR * 256),
                ("wlanAssociationAttributes", WLAN_ASSOCIATION_ATTRIBUTES),
                ("wlanSecurityAttributes", WLAN_SECURITY_ATTRIBUTES),
            ]

        data_size = wintypes.DWORD()
        data_ptr = ctypes.POINTER(WLAN_CONNECTION_ATTRIBUTES)()
        opcode = 7  # wlan_intf_opcode_current_connection
        ret = self._wlanapi.WlanQueryInterface(
            self._handle, ctypes.byref(self._guid), opcode, None,
            ctypes.byref(data_size), ctypes.byref(data_ptr), None,
        )
        if ret:
            return None
        try:
            attrs = data_ptr.contents
            if attrs.isState != 1:  # wlan_interface_state_connected
                return None
            quality = attrs.wlanAssociationAttributes.wlanSignalQuality
        finally:
            self._wlanapi.WlanFreeMemory(data_ptr)
        return quality / 2.0 - 100.0

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._wlanapi.WlanCloseHandle(self._handle, None)
            self._handle = None


class Netsh(Backend):
    """Fallback: parse signal % from ``netsh wlan show interfaces``."""

    name = "netsh"
    max_rate_hz = 3.0
    _SIGNAL_RE = re.compile(r"Signal\s*:\s*(\d+)\s*%")

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("netsh backend is Windows-only")
        if self.read() is None:
            raise RuntimeError("netsh found no connected WLAN interface")

    def read(self) -> Optional[float]:
        try:
            out = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        m = self._SIGNAL_RE.search(out)
        if not m:
            return None
        return float(m.group(1)) / 2.0 - 100.0


# --------------------------------------------------------------------------
# Android (Termux)
# --------------------------------------------------------------------------

class TermuxApi(Backend):
    """Android phone via Termux + the Termux:API add-on app.

    ``termux-wifi-connectioninfo`` returns the connected network's RSSI as
    JSON. Android's framework only refreshes RSSI every ~1-3 s, so polling
    faster than 1 Hz just returns repeated values — the rate is capped
    accordingly and detection latency on a phone is a few seconds higher
    than on a laptop.
    """

    name = "termux"
    max_rate_hz = 1.0
    COMMAND = "termux-wifi-connectioninfo"

    def __init__(self):
        if shutil.which(self.COMMAND) is None:
            raise RuntimeError(
                "termux-wifi-connectioninfo not found — run `pkg install termux-api` "
                "and install the Termux:API app from F-Droid")
        if self.read() is None:
            raise RuntimeError(
                "Termux returned no RSSI — check WiFi is connected and the "
                "Termux:API app is installed and granted permissions")

    @staticmethod
    def parse(output: str) -> Optional[float]:
        try:
            info = json.loads(output)
        except (ValueError, TypeError):
            return None
        rssi = info.get("rssi") if isinstance(info, dict) else None
        if not isinstance(rssi, (int, float)) or rssi <= -127:  # -127 = invalid
            return None
        return float(rssi)

    def read(self) -> Optional[float]:
        try:
            out = subprocess.run(
                [self.COMMAND], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        return self.parse(out)


# --------------------------------------------------------------------------
# macOS
# --------------------------------------------------------------------------

class Airport(Backend):
    """macOS: agrCtlRSSI from the airport private binary."""

    name = "airport"
    max_rate_hz = 4.0
    BINARY = ("/System/Library/PrivateFrameworks/Apple80211.framework/"
              "Versions/Current/Resources/airport")
    _RSSI_RE = re.compile(r"agrCtlRSSI:\s*(-?\d+)")

    def __init__(self):
        if not os.path.exists(self.BINARY):
            raise RuntimeError("airport binary not found")
        if self.read() is None:
            raise RuntimeError("airport reported no RSSI (not connected?)")

    def read(self) -> Optional[float]:
        try:
            out = subprocess.run(
                [self.BINARY, "-I"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        m = self._RSSI_RE.search(out)
        return float(m.group(1)) if m else None


# --------------------------------------------------------------------------
# Offline sources
# --------------------------------------------------------------------------

class Replay(Backend):
    """Replays a CSV recorded with ``wifi_sensing record`` (timestamp,rssi)."""

    name = "replay"
    max_rate_hz = 1000.0  # replay as fast as the consumer asks

    def __init__(self, path: str):
        self._rows = []
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith(("#", "timestamp")):
                    continue
                self._rows.append(float(row[-1]))
        if not self._rows:
            raise RuntimeError(f"no samples in {path}")
        self._i = 0

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._rows)

    def read(self) -> Optional[float]:
        if self.exhausted:
            return None
        v = self._rows[self._i]
        self._i += 1
        return v


class Simulate(Backend):
    """Synthetic RSSI: quiet-room noise, with human-motion bursts.

    Idle: small Gaussian jitter around a slowly drifting baseline.
    Motion: adds multipath fading — larger swings concentrated in the
    0.3-2 Hz band a walking person produces.
    """

    name = "simulate"
    max_rate_hz = 1000.0

    def __init__(self, rate_hz: float = 10.0, motion_period_s: float = 20.0,
                 motion_duty: float = 0.4, seed: Optional[int] = None):
        self.rate_hz = rate_hz
        self.motion_period_s = motion_period_s
        self.motion_duty = motion_duty
        self._rng = random.Random(seed)
        self._t = 0.0
        self._baseline = -55.0
        self._phase = self._rng.uniform(0, 2 * math.pi)

    def motion_active(self) -> bool:
        return (self._t % self.motion_period_s) < self.motion_period_s * self.motion_duty

    def read(self) -> Optional[float]:
        dt = 1.0 / self.rate_hz
        self._t += dt
        self._baseline += self._rng.gauss(0, 0.01)  # slow thermal/AGC drift
        v = self._baseline + self._rng.gauss(0, 0.4)
        if self.motion_active():
            f = 0.9  # dominant body-motion fade frequency, Hz
            self._phase += 2 * math.pi * f * dt
            v += 2.5 * math.sin(self._phase) + self._rng.gauss(0, 1.2)
        return v


# --------------------------------------------------------------------------
# Auto-detection
# --------------------------------------------------------------------------

def autodetect(interface: Optional[str] = None) -> Backend:
    """Return the most efficient working backend for this machine."""
    candidates = []
    if sys.platform.startswith("linux"):
        candidates = []
        # Termux on Android identifies as linux but has no /proc/net/wireless
        # access; prefer its API bridge when present.
        if os.environ.get("TERMUX_VERSION") or shutil.which(TermuxApi.COMMAND):
            candidates.append(TermuxApi)
        candidates += [lambda: ProcNetWireless(interface), lambda: IwLink(interface)]
    elif sys.platform == "win32":
        candidates = [WlanApi, Netsh]
    elif sys.platform == "darwin":
        candidates = [Airport]

    errors = []
    for factory in candidates:
        try:
            return factory()
        except Exception as e:  # try the next candidate
            errors.append(str(e))
    raise RuntimeError(
        "No usable RSSI source found. Are you connected to WiFi?\n  - "
        + "\n  - ".join(errors)
    )
