"""
hivelink.server
FastAPI node server:
  GET  /              — dashboard UI
  GET  /api/cluster   — cluster status
  GET  /api/hardware  — this node's hardware
  GET  /api/models    — runnable models
  GET  /api/plan/{m}  — layer assignment plan
  GET  /api/stats     — live CPU/GPU stats for this node
  POST /v1/chat/completions  — OpenAI-compatible inference
  GET  /v1/models     — OpenAI model list
  WS   /ws/cluster    — live peer + stats updates
  GET  /health        — health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .discovery import DiscoveryService, PeerInfo
from .hardware import HardwareProfile, detect as detect_hardware
from .scheduler import MODEL_SPECS, assign_layers, can_run_model
from .stats import collect as collect_stats, NodeStats

logger = logging.getLogger(__name__)

_node_id: str   = str(uuid.uuid4())
_hardware: HardwareProfile | None = None
_discovery: DiscoveryService | None = None
_ws_clients: list[WebSocket] = []
_last_stats: NodeStats | None = None

API_PORT              = int(os.environ.get("HIVELINK_PORT", "47730"))
LLAMA_CPP_SERVER_PORT = int(os.environ.get("LLAMA_CPP_PORT", "8080"))
# Static peers: comma-separated IPs, e.g. "192.168.1.112" or "192.168.1.112:47730"
_STATIC_PEERS = [p.strip() for p in os.environ.get("HIVELINK_PEERS", "").split(",") if p.strip()]
DASHBOARD_PATH        = Path(__file__).parent.parent / "dashboard" / "index.html"
STATS_INTERVAL        = 2.0   # seconds between live stats broadcasts


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hardware, _discovery
    logger.info("HiveLink node starting — id=%s", _node_id[:8])
    _hardware = detect_hardware()
    logger.info("Hardware: %s | %s | %.1f TFLOPS | %dMB VRAM",
                _hardware.primary_backend, _hardware.cpu_name,
                _hardware.total_fp16_tflops, _hardware.total_vram_mb)
    _discovery = DiscoveryService(
        node_id        = _node_id,
        api_port       = API_PORT,
        hardware       = _hardware,
        on_peer_change = _on_peer_change,
        static_peers   = _STATIC_PEERS,
    )
    if _STATIC_PEERS:
        logger.info("Static peers: %s", _STATIC_PEERS)
    peer_task  = asyncio.create_task(_discovery.start())
    stats_task = asyncio.create_task(_stats_loop())
    yield
    await _discovery.stop()
    peer_task.cancel()
    stats_task.cancel()


def _on_peer_change(peers: dict[str, PeerInfo]) -> None:
    asyncio.create_task(_broadcast_peers(peers))


async def _broadcast_peers(peers: dict[str, PeerInfo]) -> None:
    """Push cluster update to all WebSocket clients."""
    if not _ws_clients:
        return
    payload = json.dumps({
        "type":  "cluster_update",
        "peers": [p.to_dict() for p in peers.values()],
        "ts":    time.time(),
    })
    await _send_to_all(payload)


async def _fetch_peer_stats(peer: "PeerInfo") -> None:
    """Fetch stats from a remote peer and relay them to all connected dashboard clients."""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get(f"http://{peer.ip}:{peer.api_port}/api/stats")
            if resp.status_code == 200:
                payload = json.dumps({
                    "type":    "stats_update",
                    "node_id": peer.node_id,
                    "stats":   resp.json(),
                    "ts":      time.time(),
                })
                await _send_to_all(payload)
    except Exception as e:
        logger.debug("Peer stats fetch error for %s: %s", peer.node_id[:8], e)


async def _stats_loop() -> None:
    """Collect this node's stats + relay peer stats every STATS_INTERVAL seconds."""
    global _last_stats
    await asyncio.sleep(2)   # let server start up first
    while True:
        try:
            # 1. Collect and broadcast this node's own stats
            stats = await collect_stats()
            _last_stats = stats
            if _ws_clients:
                payload = json.dumps({
                    "type":    "stats_update",
                    "node_id": _node_id,
                    "stats":   stats.to_dict(),
                    "ts":      time.time(),
                })
                await _send_to_all(payload)

            # 2. Fetch and relay stats from all other peers
            if _discovery and _ws_clients:
                for peer in _discovery.active_peers():
                    if peer.node_id != _node_id:
                        asyncio.create_task(_fetch_peer_stats(peer))

        except Exception as e:
            logger.debug("Stats collection error: %s", e)
        await asyncio.sleep(STATS_INTERVAL)


async def _send_to_all(payload: str) -> None:
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


app = FastAPI(title="HiveLink", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if DASHBOARD_PATH.exists():
        return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Dashboard not found.</h2>", status_code=404)


@app.websocket("/ws/cluster")
async def ws_cluster(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    # Send current cluster state immediately on connect
    if _discovery:
        await websocket.send_text(json.dumps({
            "type":  "cluster_update",
            "peers": [p.to_dict() for p in _discovery.peers.values()],
            "ts":    time.time(),
        }))
    # Also send latest stats if available
    if _last_stats:
        await websocket.send_text(json.dumps({
            "type":    "stats_update",
            "node_id": _node_id,
            "stats":   _last_stats.to_dict(),
            "ts":      time.time(),
        }))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@app.get("/api/cluster")
async def get_cluster():
    if not _discovery:
        raise HTTPException(503, "Discovery not running")
    peers = _discovery.active_peers()
    return {
        "node_id":      _node_id,
        "peer_count":   len(peers),
        "peers":        [p.to_dict() for p in peers],
        "total_vram_mb":sum(p.memory_score_mb for p in peers),
        "total_tflops": round(sum(p.tflops for p in peers), 1),
    }


@app.get("/api/hardware")
async def get_hardware():
    if not _hardware:
        raise HTTPException(503, "Hardware not detected")
    return _hardware.to_dict()


@app.get("/api/stats")
async def get_stats():
    """Live CPU/GPU stats for this node."""
    stats = await collect_stats()
    return stats.to_dict()


async def _ollama_models() -> list[str]:
    """Fetch currently-loaded (actively serving) models from Ollama / llama-server / vLLM / MLX."""
    ports_to_try = list(dict.fromkeys([LLAMA_CPP_SERVER_PORT, 11434, 8080, 8000, 10240]))
    async with httpx.AsyncClient(timeout=2) as client:
        for port in ports_to_try:
            try:
                resp = await client.get(f"http://127.0.0.1:{port}/v1/models")
                if resp.status_code == 200:
                    ids = [m["id"] for m in resp.json().get("data", [])]
                    if ids:
                        return ids
            except Exception:
                continue
    return []


async def _ollama_cached_models() -> list[dict]:
    """
    Fetch models pulled and cached on disk via Ollama's native /api/tags endpoint.
    This is distinct from _ollama_models() above: a model can be cached (downloaded,
    ready to use) without being currently loaded into memory. /api/tags also returns
    real size_mb and parameter info straight from Ollama, which is more accurate
    than our static MODEL_SPECS estimates.
    """
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                out = []
                for m in models:
                    details = m.get("details", {}) or {}
                    out.append({
                        "model_id":      m.get("name", ""),
                        "size_mb":       round((m.get("size", 0) or 0) / (1024 * 1024)),
                        "param_size":    details.get("parameter_size", ""),   # e.g. "8.0B"
                        "quant_level":   details.get("quantization_level", ""),  # e.g. "Q4_K_M"
                        "family":        details.get("family", ""),
                        "modified_at":   m.get("modified_at", ""),
                    })
                return out
    except Exception:
        pass
    return []


@app.get("/api/models")
async def get_models():
    if not _discovery:
        raise HTTPException(503, "Discovery not running")
    peers = _discovery.active_peers()

    live_model_ids   = await _ollama_models()
    cached_models    = await _ollama_cached_models()
    live_ids_lower   = {m.lower() for m in live_model_ids}
    cached_ids_lower = {m["model_id"].lower() for m in cached_models}

    models = []

    # 1. Live models — currently loaded and actively serving requests
    for mid in live_model_ids:
        cached_match = next((c for c in cached_models if c["model_id"].lower() == mid.lower()), None)
        models.append({
            "model_id":    mid,
            "params_b":    _parse_param_size(cached_match["param_size"]) if cached_match else 0,
            "layers":      0,
            "quant_bits":  4,
            "size_mb":     cached_match["size_mb"] if cached_match else 0,
            "can_run":     True,
            "live":        True,
            "cached":      True,
        })

    # 2. Cached models — pulled and ready on disk, but not currently loaded into memory
    for c in cached_models:
        if c["model_id"].lower() in live_ids_lower:
            continue  # already listed above as live
        models.append({
            "model_id":    c["model_id"],
            "params_b":    _parse_param_size(c["param_size"]),
            "layers":      0,
            "quant_bits":  4,
            "size_mb":     c["size_mb"],
            "can_run":     True,
            "live":        False,
            "cached":      True,
        })

    # 3. Known models from MODEL_SPECS — not yet pulled anywhere, shown as pullable
    #    if the cluster's combined memory could theoretically run them
    for model_id, (layers, params_b) in MODEL_SPECS.items():
        already_shown = any(model_id in lid for lid in live_ids_lower | cached_ids_lower)
        if already_shown:
            continue
        for bits in [4, 8]:
            check = can_run_model(peers, model_id, bits)
            if check["can_run"]:
                models.append({
                    "model_id":   model_id,
                    "params_b":   params_b,
                    "layers":     layers,
                    "quant_bits": bits,
                    "size_mb":    check["needed_mb"],
                    "can_run":    True,
                    "live":       False,
                    "cached":     False,
                })
                break

    return {"models": models}


def _parse_param_size(param_size: str) -> float:
    """Parse Ollama's parameter_size string (e.g. '8.0B', '671B', '3.8M') into a float of billions."""
    if not param_size:
        return 0.0
    s = param_size.strip().upper()
    try:
        if s.endswith("B"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1000.0
        return float(s)
    except ValueError:
        return 0.0


@app.get("/api/plan/{model_id}")
async def get_plan(
    model_id:      str,
    quant_bits:    int = 4,
    sharding_mode: str = "pipeline",
    min_nodes:     int = 1,
):
    if not _discovery:
        raise HTTPException(503, "Discovery not running")
    if sharding_mode not in ("pipeline", "tensor"):
        raise HTTPException(422, f"Unknown sharding_mode '{sharding_mode}' — use 'pipeline' or 'tensor'")
    if sharding_mode == "tensor":
        raise HTTPException(
            422,
            "Tensor sharding isn't implemented yet — only pipeline sharding is supported "
            "today. This is tracked on the roadmap (v0.4)."
        )

    peers = _discovery.active_peers()
    if len(peers) < min_nodes:
        raise HTTPException(
            422,
            f"Minimum nodes not met — {len(peers)} node(s) online, {min_nodes} required."
        )

    plan = assign_layers(peers, model_id, quant_bits, sharding_mode=sharding_mode, min_nodes=min_nodes)
    if not plan:
        raise HTTPException(422, f"Cluster cannot run {model_id} at Q{quant_bits}")
    return plan.to_dict()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0


async def _find_inference_port() -> int:
    """Return the first port where an inference server is actually running."""
    ports = list(dict.fromkeys([LLAMA_CPP_SERVER_PORT, 11434, 8080, 8000, 10240]))
    async with httpx.AsyncClient(timeout=1) as client:
        for port in ports:
            try:
                r = await client.get(f"http://127.0.0.1:{port}/v1/models")
                if r.status_code == 200 and r.json().get("data"):
                    return port
            except Exception:
                continue
    return LLAMA_CPP_SERVER_PORT


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    port      = await _find_inference_port()
    llama_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload   = {
        "model":       req.model,
        "messages":    [m.model_dump() for m in req.messages],
        "stream":      req.stream,
        "temperature": req.temperature,
        "max_tokens":  req.max_tokens,
        "top_p":       req.top_p,
    }
    if req.stream:
        async def gen() -> AsyncGenerator[bytes, None]:
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", llama_url, json=payload) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except Exception:
                pass
        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(llama_url, json=payload)
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(503, "Inference server not running")


@app.get("/v1/models")
async def list_models():
    if not _discovery:
        return {"object": "list", "data": []}
    peers = _discovery.active_peers()
    return {"object": "list", "data": [
        {"id": mid, "object": "model", "owned_by": "hivelink", "created": int(time.time())}
        for mid in MODEL_SPECS
        if can_run_model(peers, mid, 4)["can_run"]
    ]}


class PullRequest(BaseModel):
    model: str


@app.post("/api/pull")
async def pull_model(req: PullRequest):
    """
    Pull a model via Ollama's native pull API and stream progress back.
    Ollama exposes POST /api/pull with stream=true, returning newline-delimited
    JSON progress objects — we just relay that stream as-is.
    """
    ollama_url = "http://127.0.0.1:11434/api/pull"

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", ollama_url,
                    json={"model": req.model, "stream": True},
                ) as resp:
                    if resp.status_code != 200:
                        yield b'{"status":"error","detail":"Ollama not reachable on port 11434"}\n'
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except httpx.ConnectError:
            yield b'{"status":"error","detail":"Ollama not running on port 11434"}\n'
        except Exception as e:
            yield (str(e)[:120] + "\n").encode()

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx etc.)
        },
    )


@app.delete("/api/models/{model_id:path}")
async def delete_model(model_id: str):
    """Remove a cached model from this node via Ollama's native delete API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.request(
                "DELETE", "http://127.0.0.1:11434/api/delete",
                json={"model": model_id},
            )
            if resp.status_code == 200:
                return {"status": "deleted", "model_id": model_id}
            raise HTTPException(resp.status_code, f"Ollama returned {resp.status_code}")
    except httpx.ConnectError:
        raise HTTPException(503, "Ollama not running on port 11434")


@app.get("/health")
async def health():
    return {"status": "ok", "node_id": _node_id[:8], "ts": time.time()}
