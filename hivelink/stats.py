"""
hivelink.stats
Live per-node stats: CPU%, RAM used, GPU util%, GPU temp, GPU wattage.
Also detects which inference engine is running (Ollama, MLX, llama-server, vLLM).
"""
from __future__ import annotations
import asyncio
import subprocess
import platform
from dataclasses import dataclass, asdict
from typing import Optional
import psutil


@dataclass
class GPUStat:
    index: int
    util_pct: float       # 0-100
    temp_c: Optional[float]   # None if unavailable
    power_w: Optional[float]  # None if unavailable


@dataclass
class NodeStats:
    cpu_pct: float
    ram_used_mb: int
    ram_total_mb: int
    gpus: list[GPUStat]
    engine: str           # "ollama" | "mlx" | "vllm" | "llama-server" | "none"
    engine_port: Optional[int]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _nvidia_stats() -> list[GPUStat]:
    """Get live NVIDIA GPU stats via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    idx   = int(parts[0])
                    util  = float(parts[1])
                    temp  = float(parts[2])
                    power = float(parts[3]) if parts[3] not in ("N/A", "[N/A]") else None
                    gpus.append(GPUStat(idx, util, temp, power))
                except (ValueError, IndexError):
                    pass
        return gpus
    except Exception:
        return []


async def _detect_engine() -> tuple[str, Optional[int]]:
    """Detect which inference engine is running and on which port."""
    import httpx
    candidates = [
        (11434, "ollama"),
        (8080,  "llama-server"),
        (8000,  "vllm"),
        (10240, "mlx"),
    ]
    async with httpx.AsyncClient(timeout=0.8) as client:
        for port, name in candidates:
            try:
                r = await client.get(f"http://127.0.0.1:{port}/v1/models")
                if r.status_code == 200:
                    data = r.json()
                    # MLX has a specific user-agent or path difference
                    if name == "llama-server":
                        # Check if it might be vLLM or MLX on 8080
                        owned_by = ""
                        if data.get("data"):
                            owned_by = data["data"][0].get("owned_by", "")
                        if "vllm" in owned_by.lower():
                            return "vllm", port
                    return name, port
            except Exception:
                continue
    # MLX can also run on 8080 — check headers
    return "none", None


async def collect() -> NodeStats:
    """Collect all live stats for this node."""
    os_name = platform.system().lower()

    # CPU + RAM (all platforms)
    cpu_pct = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    ram_used_mb  = mem.used  // (1024 * 1024)
    ram_total_mb = mem.total // (1024 * 1024)

    # GPU stats
    gpus: list[GPUStat] = []
    if os_name != "darwin":
        # NVIDIA / AMD
        gpus = _nvidia_stats()
        # AMD via rocm-smi if no NVIDIA
        if not gpus:
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showuse", "--showtemp", "--json"],
                    text=True, timeout=3,
                )
                import json
                data = json.loads(out)
                for i, (_, card) in enumerate(data.items()):
                    util  = float(card.get("GPU use (%)", 0))
                    temp  = float(card.get("Temperature (Sensor edge) (C)", 0))
                    gpus.append(GPUStat(i, util, temp, None))
            except Exception:
                pass
    else:
        # Apple Silicon — no GPU util/temp without sudo powermetrics
        # Show 0% util as placeholder
        gpus = [GPUStat(0, 0.0, None, None)]

    # Engine detection
    engine, engine_port = await _detect_engine()

    return NodeStats(
        cpu_pct      = round(cpu_pct, 1),
        ram_used_mb  = ram_used_mb,
        ram_total_mb = ram_total_mb,
        gpus         = gpus,
        engine       = engine,
        engine_port  = engine_port,
    )
