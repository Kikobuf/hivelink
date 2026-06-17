"""
hivelink.server
FastAPI node server:
  GET  /              — dashboard UI
  GET  /api/cluster   — cluster status
  GET  /api/hardware  — this node's hardware
  GET  /api/models    — runnable models
  GET  /api/plan/{m}  — layer assignment plan
  POST /v1/chat/completions  — OpenAI-compatible inference
  GET  /v1/models     — OpenAI model list
  WS   /ws/cluster    — live peer updates
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

logger = logging.getLogger(__name__)

_node_id: str = str(uuid.uuid4())
_hardware: HardwareProfile | None = None
_discovery: DiscoveryService | None = None
_ws_clients: list[WebSocket] = []

API_PORT             = int(os.environ.get("HIVELINK_PORT", "47730"))
LLAMA_CPP_SERVER_PORT= int(os.environ.get("LLAMA_CPP_PORT", "8080"))
DASHBOARD_PATH       = Path(__file__).parent.parent / "dashboard" / "index.html"


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
    )
    task = asyncio.create_task(_discovery.start())
    yield
    await _discovery.stop()
    task.cancel()


def _on_peer_change(peers: dict[str, PeerInfo]) -> None:
    asyncio.create_task(_broadcast(peers))


async def _broadcast(peers: dict[str, PeerInfo]) -> None:
    if not _ws_clients:
        return
    payload = json.dumps({
        "type":  "cluster_update",
        "peers": [p.to_dict() for p in peers.values()],
        "ts":    time.time(),
    })
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


app = FastAPI(title="HiveLink", version="0.1.0", lifespan=lifespan)
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
    if _discovery:
        await websocket.send_text(json.dumps({
            "type":  "cluster_update",
            "peers": [p.to_dict() for p in _discovery.peers.values()],
            "ts":    time.time(),
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


async def _ollama_models() -> list[dict]:
    """Fetch models from Ollama or llama-server.
    Auto-tries port 11434 (Ollama default) and 8080 (llama-server default)
    in addition to LLAMA_CPP_PORT, so no manual env var needed for Ollama.
    """
    ports_to_try = list(dict.fromkeys([LLAMA_CPP_SERVER_PORT, 11434, 8080]))
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


@app.get("/api/models")
async def get_models():
    if not _discovery:
        raise HTTPException(503, "Discovery not running")
    peers  = _discovery.active_peers()

    # Live models from Ollama / llama-server (what's actually running)
    live_model_ids = await _ollama_models()

    models = []

    # Add live models first — these are actually usable right now
    for mid in live_model_ids:
        models.append({
            "model_id":  mid,
            "params_b":  0,
            "layers":    0,
            "quant_bits": 4,
            "size_mb":   0,
            "can_run":   True,
            "live":      True,   # actually loaded in inference engine
        })

    # Add cluster-feasible models from the known spec list
    live_ids_lower = {m.lower() for m in live_model_ids}
    for model_id, (layers, params_b) in MODEL_SPECS.items():
        # Skip if already listed as a live model
        if any(model_id in lid for lid in live_ids_lower):
            continue
        for bits in [4, 8]:
            check = can_run_model(peers, model_id, bits)
            if check["can_run"]:
                models.append({
                    "model_id":  model_id,
                    "params_b":  params_b,
                    "layers":    layers,
                    "quant_bits":bits,
                    "size_mb":   check["needed_mb"],
                    "can_run":   True,
                    "live":      False,
                })
                break

    return {"models": models}


@app.get("/api/plan/{model_id}")
async def get_plan(model_id: str, quant_bits: int = 4):
    if not _discovery:
        raise HTTPException(503, "Discovery not running")
    peers = _discovery.active_peers()
    plan  = assign_layers(peers, model_id, quant_bits)
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


async def _find_inference_port() -> int:
    """Return the first port where an inference server is actually running."""
    ports = list(dict.fromkeys([LLAMA_CPP_SERVER_PORT, 11434, 8080]))
    async with httpx.AsyncClient(timeout=1) as client:
        for port in ports:
            try:
                r = await client.get(f"http://127.0.0.1:{port}/v1/models")
                if r.status_code == 200 and r.json().get("data"):
                    return port
            except Exception:
                continue
    return LLAMA_CPP_SERVER_PORT  # fallback


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    port = await _find_inference_port()
    llama_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload   = {
        "model":       req.model,
        "messages":    [m.model_dump() for m in req.messages],
        "stream":      req.stream,
        "temperature": req.temperature,
        "max_tokens":  req.max_tokens,
    }
    if req.stream:
        async def gen() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", llama_url, json=payload) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(llama_url, json=payload)
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(503, "llama-server not running. Start it with: llama-server --model <path/to/model.gguf>")


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


@app.get("/health")
async def health():
    return {"status": "ok", "node_id": _node_id[:8], "ts": time.time()}
