"""
hivelink.scheduler
Assigns model layers to cluster nodes proportional to VRAM x compute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .discovery import PeerInfo


@dataclass
class LayerAssignment:
    node_id: str
    api_url: str
    layer_start: int
    layer_end: int
    backend: str
    vram_mb: int

    @property
    def layer_count(self) -> int:
        return self.layer_end - self.layer_start + 1

    def to_dict(self) -> dict:
        return {
            "node_id":     self.node_id,
            "api_url":     self.api_url,
            "layer_start": self.layer_start,
            "layer_end":   self.layer_end,
            "layer_count": self.layer_count,
            "backend":     self.backend,
            "vram_mb":     self.vram_mb,
        }


@dataclass
class ClusterPlan:
    model_id: str
    total_layers: int
    total_params_b: float
    quant_bits: int
    assignments: list[LayerAssignment]

    @property
    def node_count(self) -> int:
        return len(self.assignments)

    def to_dict(self) -> dict:
        return {
            "model_id":      self.model_id,
            "total_layers":  self.total_layers,
            "total_params_b":self.total_params_b,
            "quant_bits":    self.quant_bits,
            "node_count":    self.node_count,
            "assignments":   [a.to_dict() for a in self.assignments],
        }


# (total_layers, params_B)
MODEL_SPECS: dict[str, tuple[int, float]] = {
    "llama3-8b":        (32,   8.0),
    "llama3-70b":       (80,  70.0),
    "llama3-405b":      (126, 405.0),
    "qwen2.5-7b":       (28,   7.0),
    "qwen2.5-14b":      (48,  14.0),
    "qwen2.5-32b":      (64,  32.0),
    "qwen2.5-72b":      (80,  72.0),
    "mistral-7b":       (32,   7.0),
    "mixtral-8x7b":     (32,  46.7),
    "deepseek-r1-7b":   (28,   7.0),
    "deepseek-r1-70b":  (80,  70.0),
    "gemma2-9b":        (42,   9.0),
    "gemma2-27b":       (46,  27.0),
}


def model_size_mb(params_b: float, quant_bits: int) -> float:
    return params_b * 1e9 * (quant_bits / 8.0) / (1024 * 1024)


def assign_layers(
    peers: list[PeerInfo],
    model_id: str,
    quant_bits: int = 4,
    custom_layers: int | None = None,
    custom_params_b: float | None = None,
) -> ClusterPlan | None:
    if not peers:
        return None

    spec = MODEL_SPECS.get(model_id.lower())
    if spec:
        total_layers, params_b = spec
    else:
        total_layers = custom_layers or 32
        params_b     = custom_params_b or 7.0

    total_size_mb  = model_size_mb(params_b, quant_bits)
    total_memory   = sum(p.memory_score_mb for p in peers)

    if total_memory < total_size_mb * 0.95:
        return None

    weights = []
    for peer in peers:
        mem     = min(peer.memory_score_mb, total_size_mb)
        compute = max(peer.tflops, 0.5)
        weights.append(math.sqrt(mem) * math.log1p(compute))

    total_weight  = sum(weights)
    assignments   = []
    layer_cursor  = 0

    for i, (peer, weight) in enumerate(zip(peers, weights)):
        if i == len(peers) - 1:
            n_layers = total_layers - layer_cursor
        else:
            n_layers = max(1, round(total_layers * weight / total_weight))

        n_layers = min(n_layers, total_layers - layer_cursor)
        if n_layers <= 0:
            break

        layer_end = layer_cursor + n_layers - 1
        layer_mb  = int(total_size_mb * n_layers / total_layers)
        backend   = peer.hardware.get("primary_backend", "cpu")

        assignments.append(LayerAssignment(
            node_id     = peer.node_id,
            api_url     = peer.api_url,
            layer_start = layer_cursor,
            layer_end   = layer_end,
            backend     = backend,
            vram_mb     = layer_mb,
        ))
        layer_cursor = layer_end + 1
        if layer_cursor >= total_layers:
            break

    return ClusterPlan(
        model_id       = model_id,
        total_layers   = total_layers,
        total_params_b = params_b,
        quant_bits     = quant_bits,
        assignments    = assignments,
    )


def can_run_model(peers: list[PeerInfo], model_id: str, quant_bits: int = 4) -> dict:
    spec = MODEL_SPECS.get(model_id.lower())
    if not spec:
        return {"can_run": False, "reason": "Unknown model"}

    total_layers, params_b = spec
    needed_mb    = model_size_mb(params_b, quant_bits)
    available_mb = sum(p.memory_score_mb for p in peers)

    return {
        "can_run":      available_mb >= needed_mb * 0.95,
        "needed_mb":    int(needed_mb),
        "available_mb": available_mb,
        "deficit_mb":   max(0, int(needed_mb - available_mb)),
        "params_b":     params_b,
        "quant_bits":   quant_bits,
    }
