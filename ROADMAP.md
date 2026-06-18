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

## ✅ v0.2 — Live stats

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

## 🔨 v0.3 — Model management (next)

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

## 🔨 v0.8 — Multi-engine support (per-node engine choice)

> Goal: let each node run whichever engine the user prefers — Ollama, MLX, vLLM, llama-server — and pick which one to use per-model from the dashboard. Cross-node *splitting* stays on llama.cpp/GGUF (the only format that works the same across CUDA/Metal/ROCm); this is about choosing a full standalone model+engine per node, not cross-engine layer splitting.

- [ ] Model dropdown shows engine alongside model name — e.g. "llama3.2 (MLX · Mac mini)" vs "llama3.2 (Ollama · Windows)"
- [ ] Dashboard lets you pick which node/engine handles a given chat request when not using cluster-split mode
- [ ] MLX install guide for Mac — `pip install mlx-lm`, run `mlx_lm.server`, HiveLink auto-detects via existing engine detection
- [ ] vLLM install guide for NVIDIA — `pip install vllm`, run `vllm serve`, HiveLink auto-detects via existing engine detection
- [ ] Document the constraint clearly in UI: cross-node layer splitting requires llama.cpp/GGUF on all participating nodes; MLX/vLLM nodes serve standalone (non-split) models only, selectable individually
- [ ] (Stretch, no committed timeline) Investigate cross-engine pipeline splitting — different tensor formats and activation protocols make this a hard problem, likely its own research spike rather than a quick feature

---

## 🔨 v0.9 — Tool calling + MCP support

> Goal: let local models use tools — web search, file access, custom MCP servers — the same way Claude Desktop does, but running on your own cluster.

- [ ] OpenAI-compatible tool calling — extend `/v1/chat/completions` to accept and return `tools` / `tool_calls`
- [ ] "Tool-capable" badge on models in dashboard — not all local models handle function calling reliably (Llama 3.x, Qwen2.5, Mistral are solid; smaller/older models often hallucinate calls)
- [ ] MCP bridge — HiveLink connects to MCP servers (stdio/HTTP), forwards tool calls from the model, returns results for the next turn
- [ ] New "Tools" tab in dashboard — connect/manage MCP servers (web search, filesystem, custom servers like KB Rides Shopify MCP)
- [ ] Tool-call indicator in chat — "Calling web_search…" animation similar to the thinking-dots display
- [ ] Note: tool-calling reliability scales with model size — this is a natural pull toward running larger models (Qwen2.5-32B+) across the cluster, which is exactly what HiveLink is for

---

## 🔨 v0.10 — `hivelink launch <tool>` (agent tool integrations)

> Goal: copy Ollama's `ollama launch <tool>` pattern — one command installs, configures, and starts an agent tool already wired to your HiveLink cluster instead of a single machine. ([Reference: Ollama's integration docs](https://docs.ollama.com/integrations))

- [ ] **Anthropic-compatible API shim** — translate Claude's Messages API shape to/from HiveLink's existing OpenAI-compatible backend, so `ANTHROPIC_BASE_URL=http://localhost:47730` works directly (Ollama already proved this is solvable: see their `claude-code` integration)
- [ ] `hivelink launch claude` — installs Claude Code if missing, model picker pulls from cluster's live models, sets `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` automatically, launches
- [ ] `hivelink launch openclaw` — same pattern for OpenClaw (directly relevant — this is what James/Kimi K2 already runs); model picker, gateway config, launch
- [ ] Context-length guard — both Claude Code and OpenClaw need ~64k+ context to behave well as agents; warn in the model picker if a selected model's context window is too small
- [ ] `hivelink launch --model <name>` — skip the picker, launch directly with a named cluster model (mirrors Ollama's `--model` flag)
- [ ] Document recommended model sizes for agentic use (32B+ tends to be the practical floor for reliable tool use, matching Ollama's own recommendations for these integrations)
- [ ] Stretch: extend the same pattern to other agent tools as they gain Anthropic/OpenAI-compatible support (Codex, Goose, Cline, etc.)

---

## 🔬 v0.11 — Cross-engine pipeline splitting (research spike, not committed)

> Goal: investigate whether a single model's layers could be split across *different* inference engines on different nodes (e.g. vLLM layers on Windows + MLX layers on Mac for the same model) — not promised as a shippable feature, since no existing project has solved this cleanly.

**Why this is hard, stated plainly:**
- **Incompatible weight formats** — GGUF (llama.cpp), safetensors (vLLM), and MLX's own quantized format are different on-disk representations; there's no clean shared format to split a model across them without per-node conversion at load time
- **Incompatible activation handoff** — HiveLink's current pipeline splitting works because all llama.cpp nodes share the same tensor layout for passing activations between layers; vLLM and MLX have different internal tensor shapes/dtypes at layer boundaries, so a translation layer would need to be built from scratch
- **No prior art** — EXO doesn't do this; nothing in the open-source distributed-inference space currently does this across engine families

**Scoped as a spike, not a feature:**
- [ ] Prototype a single-layer-boundary handoff between llama.cpp and one other engine (likely MLX, since Apple Silicon nodes are the most natural pairing) — purely to test feasibility
- [ ] Measure conversion overhead per activation handoff — if it's too slow to be worth it, document why and close this out
- [ ] Decision point after the prototype: pursue as a real feature, or formally mark as "not pursuing" with reasoning recorded here so it doesn't get re-asked every few months
- [ ] If pursued: would need its own dedicated milestone (v1.x), this spike is scoping only

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
