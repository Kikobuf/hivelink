# HiveLink

**Run large AI models across your mixed-hardware devices — Mac, Windows, Linux, NVIDIA, AMD, Apple Silicon.**

HiveLink is a cross-platform distributed LLM inference system. It pools the memory and compute of every machine on your LAN into a single AI cluster, then automatically splits model layers across them so you can run models that don't fit on any single device.

Think of it as EXO — but it actually works on Windows and NVIDIA.

---

## How it works

HiveLink uses **pipeline parallelism**: it splits a model's transformer layers across your devices, so each machine processes its assigned chunk and passes the result to the next. The split is weighted by each device's VRAM and compute — your RTX 3080 Ti gets more layers than a CPU-only node.

```
┌─────────────────────────────────────────────────────────────────┐
│  Llama 3 70B — Q4 — split across 2 nodes                       │
├─────────────────────────────────────┬───────────────────────────┤
│  Windows · RTX 3080 Ti (12GB CUDA)  │  Mac mini · M2 Pro (Metal)│
│  Layers 0–31  (40%)                 │  Layers 32–79  (60%)      │
└─────────────────────────────────────┴───────────────────────────┘
         ↑ token in                            ↓ token out
```

All nodes use **llama.cpp** as the inference backend — the same GGUF model file works on every OS and GPU. No format conversion between nodes, so performance is close to single-device.

---

## Supported hardware

| Platform            | OS            | GPU                  | Backend   |
|---------------------|---------------|----------------------|-----------|
| Apple Silicon Mac   | macOS 12+     | Unified memory (any) | Metal     |
| Windows PC          | Windows 10/11 | NVIDIA RTX / GTX     | CUDA      |
| Linux workstation   | Ubuntu 22+    | NVIDIA / AMD         | CUDA/ROCm |
| Any machine         | Any           | None needed          | CPU       |
| Raspberry Pi / SBC  | Linux (ARM)   | None needed          | CPU       |

---

## Quick start

```bash
# 1. Install (same command on all platforms)
pip install hivelink

# With NVIDIA GPU
pip install "hivelink[nvidia]"

# 2. Install llama.cpp backend
hivelink install-llama

# 3. Start the node daemon on EVERY machine
hivelink start

# 4. Check your cluster (from any node)
hivelink status
hivelink models
hivelink plan llama3-70b
```

Nodes discover each other automatically via UDP broadcast — no config needed as long as they're on the same LAN.

---

## Dashboard

Open `http://localhost:47730` (or `http://localhost:47730/dashboard`) in your browser to see:

- All discovered nodes with backend type, memory, and TFLOPS
- Models your cluster can run
- Live layer distribution visualization — shows exactly which device handles which layers
- OpenAI-compatible API reference

---

## OpenAI-compatible API

HiveLink exposes a drop-in replacement for the OpenAI API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:47730/v1",
    api_key="hivelink",  # any value
)

response = client.chat.completions.create(
    model="llama3-70b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    print(chunk.choices.delta.content, end="")
```

Works with any OpenAI-compatible client: LangChain, LlamaIndex, Open WebUI, SillyTavern, etc.

---

## CLI reference

```
hivelink start          Start the node daemon
hivelink status         Show all cluster nodes
hivelink models         List models the cluster can run
hivelink plan <model>   Show layer distribution plan
hivelink hardware       Detect this machine's hardware
hivelink install-llama  Download llama.cpp server binary
```

---

## Supported models

| Model           | Params | Layers |
|-----------------|--------|--------|
| llama3-8b       | 8B     | 32     |
| llama3-70b      | 70B    | 80     |
| llama3-405b     | 405B   | 126    |
| qwen2.5-7b      | 7B     | 28     |
| qwen2.5-14b     | 14B    | 48     |
| qwen2.5-32b     | 32B    | 64     |
| qwen2.5-72b     | 72B    | 80     |
| mistral-7b      | 7B     | 32     |
| deepseek-r1-70b | 70B    | 80     |
| gemma2-27b      | 27B    | 46     |

Any GGUF model from Hugging Face works — the known models list just enables memory feasibility checking.

---

## Architecture

```
hivelink/
├── hardware.py     Hardware detection (CUDA/Metal/ROCm/CPU, VRAM, TFLOPS)
├── discovery.py    UDP broadcast peer discovery — zero config
├── scheduler.py    Layer assignment algorithm — weighted by VRAM × compute
├── server.py       FastAPI server — REST API, WebSocket, OpenAI-compatible
└── cli.py          Typer CLI

dashboard/
└── index.html      Self-contained dashboard — no build step
```

---

## Why not just use EXO?

EXO is Mac-first. Windows isn't supported. NVIDIA GPU detection is buggy (GPUs register as 0 TFLOPS). The mixed-backend approach (MLX + tinygrad) causes 90% throughput loss when crossing node types.

HiveLink uses llama.cpp everywhere — the same GGUF format, same binary protocol for activations, full CUDA/ROCm/Metal support with zero conversion overhead.

---

## Roadmap

- [ ] Automatic model download from Hugging Face Hub
- [ ] Tensor parallelism for same-LAN high-bandwidth setups
- [ ] Web UI chat interface (not just API)
- [ ] Docker image for easy deployment
- [ ] Tailscale support for cross-network clusters
- [ ] Windows installer (.exe)
