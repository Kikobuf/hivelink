# HiveLink Roadmap

> Cross-platform distributed LLM inference — pool Mac, Windows, Linux, NVIDIA, AMD, and Apple Silicon into one AI cluster.

---

## ✅ v0.1 — Done

- [x] UDP broadcast peer discovery — zero config, nodes find each other automatically
- [x] Hardware detection — NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), CPU fallback
- [x] Layer distribution scheduler — assigns model layers proportionally by VRAM × compute
- [x] FastAPI server with OpenAI-compatible `/v1/chat/completions` endpoint
- [x] WebSocket live cluster updates
- [x] Dashboard — EXO-style node cards, hardware tab, model cards with brand logos
- [x] Chat tab — streaming responses, thinking animation, chain-of-thought display
- [x] Conversation history — localStorage, auto-save, restore past chats
- [x] Config panel — temperature, max tokens, top P, system prompt
- [x] Ollama integration — auto-detects live models, correct model names in dropdown
- [x] Real brand icons — NVIDIA, AMD, Intel, Apple, Meta, Mistral, DeepSeek, Qwen, Gemini
- [x] CLI — `hivelink start`, `status`, `models`, `plan`, `hardware`
- [x] README with Ethernet / direct connection setup guide
- [x] MIT license, public GitHub repo

---

## 🔨 v0.2 — Live stats (next)

> Goal: make the cluster view feel alive like EXO — real-time GPU%, temp, wattage per node.

- [ ] `/api/stats` endpoint — polls `nvidia-smi` (GPU util%, temp, power draw) on NVIDIA nodes, `psutil` (CPU%, RAM used) on all nodes every 2 seconds
- [ ] Apple Silicon nodes: CPU% + RAM via `psutil` only for now (GPU temp/wattage added in v0.5 native app via `powermetrics`)
- [ ] WebSocket pushes stats alongside peer updates — no separate polling needed
- [ ] Node cards show live GPU utilization bar, temperature, wattage
- [ ] Inference activity indicator — node card pulses when generation is active
- [ ] Stats history — sparkline charts (last 60s) per node in hardware tab
- [ ] Inference engine detection — auto-detect whether each node is running Ollama, MLX (`mlx_lm.server`), llama-server, or vLLM; show engine label in node card and hardware tab
- [ ] MLX support note: install `mlx_lm` on Mac mini for significantly faster Apple Silicon inference — already OpenAI-compatible, zero code changes needed
- [ ] vLLM support note: install `vllm` on Windows for higher NVIDIA throughput — already OpenAI-compatible, zero code changes needed

---

## 🔨 v0.3 — Model management

> Goal: `hivelink pull llama3-70b` just works, no manual file hunting.

- [ ] `hivelink pull <model>` — downloads GGUF from Hugging Face Hub with progress bar
- [ ] Model file streaming — store model on coordinator, nodes pull only their assigned layer slice over LAN (eliminates need to copy model to every machine)
- [ ] Model cache management — `hivelink models list`, `hivelink models remove <model>`
- [ ] Auto-detect locally cached Ollama and LM Studio models
- [ ] Model card shows download button in dashboard if not yet cached

---

## 🔨 v0.4 — Sharding controls + instances

> Goal: match EXO's instance/sharding UI, add tensor parallelism option.

- [ ] Sharding mode selector — Pipeline (current) vs Tensor (splits matrix ops, needs high bandwidth)
- [ ] Minimum nodes setting — only launch inference if at least N nodes are available
- [ ] Instance management — run multiple models simultaneously on different node subsets
- [ ] Instance panel in dashboard — launch, monitor, and kill model instances
- [ ] Auto-select sharding mode based on connection type (Ethernet vs WiFi vs Thunderbolt)

---

## 🔨 v0.5 — Native installers

> Goal: one-click install on every platform, no terminal required for end users.

- [ ] **Windows** — `.exe` installer (PyInstaller + NSIS), system tray icon, WebView2 app window
- [ ] **macOS** — `.dmg` with `.app` bundle (PyInstaller + Swift menu bar wrapper), WKWebView window, LaunchAgent auto-start
- [ ] **Linux** — AppImage (any distro, no install), `.deb` for Ubuntu/Debian, `.rpm` for Fedora
- [ ] Auto-updater — checks GitHub releases, downloads and applies updates in background
- [ ] First-run setup wizard — detects hardware, installs llama.cpp, downloads a starter model

---

## 🔨 v0.6 — Vision + file uploads

> Goal: multimodal support in the chat tab.

- [ ] Auto-detect vision-capable models (llama3.2-vision, Qwen2-VL, LLaVA, etc.)
- [ ] File upload button in chat — only shown when selected model supports vision
- [ ] Image preview in chat bubbles
- [ ] PDF upload — extract text, send as context
- [ ] Drag-and-drop into chat input

---

## 🔨 v0.7 — Cross-network clusters

> Goal: connect nodes that aren't on the same LAN.

- [ ] Tailscale integration — `hivelink join --tailscale` auto-configures discovery over Tailscale subnet
- [ ] Manual peer addition — `hivelink peer add <ip>:<port>` for static setups
- [ ] Encrypted transport — TLS between nodes for cross-network inference
- [ ] Latency-aware scheduling — deprioritize high-latency nodes for layer assignments

---

## 💡 Future ideas (unscheduled)

- **watchOS / Android companion** — cluster status glanceable on wrist or phone
- **Web-hosted dashboard** — shareable cluster status page (read-only, no inference)
- **Plugin system** — custom backends beyond llama.cpp (e.g. vLLM, TensorRT-LLM)
- **Benchmark mode** — `hivelink bench` runs standard prompts and reports tok/s per node
- **Power mode** — throttle nodes to a wattage budget (useful for running overnight)
- **PyPI release** — `pip install hivelink` from the public package index

---

## Architecture notes for contributors

```
hivelink/
├── hardware.py     Hardware detection (CUDA/Metal/ROCm/CPU, VRAM, TFLOPS)
├── discovery.py    UDP broadcast peer discovery — zero config
├── scheduler.py    Layer assignment — weighted by VRAM × compute
├── server.py       FastAPI — REST, WebSocket, OpenAI-compatible API
└── cli.py          Typer CLI

dashboard/
└── index.html      Single-file dashboard — no build step required
```

**Key design principles:**
- Every node runs the same binary regardless of OS or GPU
- GGUF model format everywhere — works on CUDA, Metal, ROCm, and CPU identically
- No format conversion between nodes — activations stay as raw tensors
- Dashboard is a single HTML file — no npm, no build step, works offline
