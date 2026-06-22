"""
hivelink.hardware
Detects available compute: GPU type, VRAM, RAM, estimated FLOPS.
Works on Windows, Linux, macOS — NVIDIA, AMD, Apple Silicon, CPU-only.
"""

from __future__ import annotations

import platform
import subprocess
import warnings
from dataclasses import dataclass, field, asdict
from typing import Literal

import psutil

BackendType = Literal["cuda", "metal", "rocm", "vulkan", "cpu"]
ConnectionType = Literal["ethernet", "wifi", "thunderbolt", "unknown"]


@dataclass
class GPUInfo:
    index: int
    name: str
    backend: BackendType
    vram_mb: int
    fp16_tflops: float


def detect_connection_type() -> "ConnectionType":
    """
    Infer the primary LAN connection type for this node using psutil's net_if_stats().

    Logic:
      - Match the network interface whose IP is our real LAN IP (same filtering as discovery.py).
      - Thunderbolt: adapter name contains "thunderbolt" or "tb" (Windows/Mac naming).
      - Ethernet: speed >= 1000 Mbps AND full duplex — characteristic of wired GbE+.
      - WiFi: speed < 1000 Mbps OR half duplex, OR interface name contains "wi-fi"/"wlan"/"wireless".
      - Unknown: psutil not available or no real LAN interface found.

    This is best-effort: virtualised environments or unusual NIC names may mis-classify.
    The result is surfaced in the dashboard for info and used to inform the auto-sharding
    suggestion (pipeline is always safe; tensor needs high-bandwidth, so Ethernet/Thunderbolt
    are the only reasonable choices when it's eventually implemented).
    """
    try:
        import socket as _socket
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()

        # Build a map of  ip -> interface_name  for all real LAN IPs
        _VIRTUAL = (
            "192.168.56.", "192.168.99.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
            "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
            "169.254.",
        )
        real_ifaces: list[str] = []
        for iface, iface_addrs in addrs.items():
            for addr in iface_addrs:
                if addr.family == _socket.AF_INET:
                    ip = addr.address
                    if ip.startswith("127.") or any(ip.startswith(p) for p in _VIRTUAL):
                        continue
                    real_ifaces.append(iface)

        if not real_ifaces:
            return "unknown"

        # Score each real interface — prefer the fastest one
        best: tuple[int, str] | None = None   # (speed_mbps, iface_name)
        for iface in real_ifaces:
            s = stats.get(iface)
            if s is None:
                continue
            speed = s.speed or 0
            if best is None or speed > best[0]:
                best = (speed, iface)

        if best is None:
            return "unknown"

        speed_mbps, iface_name = best
        name_lower = iface_name.lower()

        # Thunderbolt — explicit name match (Windows: "Thunderbolt", Mac: "tb0"/"bridge100" etc.)
        if "thunderbolt" in name_lower or (name_lower.startswith("tb") and name_lower[2:].isdigit()):
            return "thunderbolt"

        # WiFi — name-based first (most reliable cross-platform signal)
        if any(k in name_lower for k in ("wi-fi", "wifi", "wlan", "wireless", "airport", "wlp", "wlo")):
            return "wifi"

        # Speed+duplex based classification for everything else
        st = stats.get(iface_name)
        is_full_duplex = st.duplex == psutil.NIC_DUPLEX_FULL if st else False

        if speed_mbps >= 1000 and is_full_duplex:
            return "ethernet"
        elif speed_mbps > 0:
            # Low speed or half-duplex — likely WiFi even without a telltale name
            return "wifi"

        return "unknown"

    except Exception:
        return "unknown"


@dataclass
class HardwareProfile:
    node_os: str
    cpu_name: str
    cpu_cores: int
    ram_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)
    primary_backend: BackendType = "cpu"
    total_vram_mb: int = 0
    total_fp16_tflops: float = 0.0
    connection_type: ConnectionType = "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpus"] = [asdict(g) for g in self.gpus]
        return d

    @property
    def memory_score_mb(self) -> int:
        if self.total_vram_mb > 0:
            return self.total_vram_mb
        return self.ram_mb


def _detect_nvidia() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynvml import (  # type: ignore
                nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown,
            )
        nvmlInit()
        count = nvmlDeviceGetCount()
        for i in range(count):
            h = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = nvmlDeviceGetMemoryInfo(h)
            vram_mb = mem.total // (1024 * 1024)
            gpus.append(GPUInfo(i, name, "cuda", vram_mb, _nvidia_tflops(name)))
        nvmlShutdown()
        return gpus
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        for i, line in enumerate(out.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                name, vram_mb = parts[0], int(parts[1])
                gpus.append(GPUInfo(i, name, "cuda", vram_mb, _nvidia_tflops(name)))
    except Exception:
        pass
    return gpus


def _nvidia_tflops(name: str) -> float:
    n = name.lower()
    table = {
        "5090": 209.0, "5080": 137.0, "5070 ti": 74.0, "5070": 61.0,
        "4090": 165.0, "4080": 97.0,  "4070 ti": 80.0, "4070": 56.0,
        "3090": 71.0,  "3080 ti": 64.0, "3080": 59.0, "3070 ti": 43.0,
        "3070": 40.0,  "3060 ti": 32.0, "3060": 25.0,
        "a100": 312.0, "h100": 989.0,   "l40": 362.0,
    }
    for key, t in table.items():
        if key in n:
            return t
    return 10.0


def _detect_amd() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    try:
        import pyamdgpuinfo  # type: ignore
        count = pyamdgpuinfo.detect_gpus()
        for i in range(count):
            gpu = pyamdgpuinfo.get_gpu(i)
            vram_mb = gpu.memory_info["vram_size"] // (1024 * 1024)
            gpus.append(GPUInfo(i, gpu.name, "rocm", vram_mb, _amd_tflops(gpu.name)))
        return gpus
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            text=True, timeout=5,
        )
        import json
        data = json.loads(out)
        for i, (_, card) in enumerate(data.items()):
            name = card.get("Card series", "AMD GPU")
            vram_bytes = int(card.get("VRAM Total Memory (B)", 0))
            gpus.append(GPUInfo(i, name, "rocm", vram_bytes // (1024 * 1024), _amd_tflops(name)))
    except Exception:
        pass
    return gpus


def _amd_tflops(name: str) -> float:
    n = name.lower()
    table = {
        "7900 xtx": 122.8, "7900 xt": 103.0, "7800 xt": 74.6,
        "7700 xt": 54.9,   "6900 xt": 46.1,   "6800 xt": 41.0,
    }
    for key, t in table.items():
        if key in n:
            return t
    return 8.0


def _detect_apple() -> list[GPUInfo]:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True, timeout=5
        )
        chip = "Apple Silicon"
        for line in out.splitlines():
            if "Chip" in line or "Processor" in line:
                chip = line.split(":")[-1].strip()
                break
        ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        return [GPUInfo(0, chip, "metal", ram_mb, _apple_tflops(chip))]
    except Exception:
        return []


def _apple_tflops(chip: str) -> float:
    c = chip.lower()
    table = {
        "m4 ultra": 274.0, "m4 max": 68.0, "m4 pro": 54.0, "m4": 38.0,
        "m3 ultra": 256.0, "m3 max": 60.0, "m3 pro": 42.0, "m3": 35.0,
        "m2 ultra": 220.0, "m2 max": 54.0, "m2 pro": 41.0, "m2": 15.8,
        "m1 ultra": 192.0, "m1 max": 43.0, "m1 pro": 32.0, "m1": 10.4,
    }
    for key, t in table.items():
        if key in c:
            return t
    return 10.0


def detect() -> HardwareProfile:
    os_name = platform.system().lower()
    cpu_name = platform.processor() or platform.machine()
    cpu_cores = psutil.cpu_count(logical=False) or 1
    ram_mb = psutil.virtual_memory().total // (1024 * 1024)

    gpus: list[GPUInfo] = []
    primary_backend: BackendType = "cpu"

    if os_name == "darwin" and platform.machine() == "arm64":
        gpus = _detect_apple()
        if gpus:
            primary_backend = "metal"
    else:
        nvidia = _detect_nvidia()
        if nvidia:
            gpus, primary_backend = nvidia, "cuda"
        else:
            amd = _detect_amd()
            if amd:
                gpus, primary_backend = amd, "rocm"

    total_vram = sum(g.vram_mb for g in gpus)
    total_tflops = sum(g.fp16_tflops for g in gpus)
    conn_type = detect_connection_type()

    return HardwareProfile(
        node_os=os_name,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_mb=ram_mb,
        gpus=gpus,
        primary_backend=primary_backend,
        total_vram_mb=total_vram,
        total_fp16_tflops=total_tflops,
        connection_type=conn_type,
    )
