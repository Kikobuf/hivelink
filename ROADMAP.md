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

- [x] `/api/stats` endpoint — polls `nvidia-smi` (GPU util%, temp, power draw) on NVIDIA nodes, `psutil` (CPU%, RAM used) on all nodes every 2 seconds
- [x] Apple Silicon nodes: CPU% + RAM via `psutil` only for now (GPU temp/wattage added in v0.5 native app via `powermetrics`)
- [x] WebSocket pushes stats alongside peer updates — no separate polling needed
- [x] Node cards show live GPU utilization bar, temperature, wattage
- [x] Inference activity indicator — node card pulses when generation is active
- [x] Stats history — sparkline charts (last 60s) per node in hardware tab
- [x] Inference engine detection — auto-detect whether each node is running Ollama, MLX (`mlx_lm.server`), llama-server, or vLLM; show engine label in node card and hardware tab

---

## ✅ v0.3 — Model management (done)

> Goal: `hivelink pull llama3.2` just works, no manual file hunting.

- [x] `hivelink pull <model>` — CLI command, proxies to `ollama pull` with live progress
- [x] Dashboard "Pull model" button — modal with live download progress bar, streamed from Ollama's native pull API
- [x] Auto-detect cached Ollama models — `/api/tags` integration, dashboard shows live / cached / pullable states distinctly
- [x] Model cache management — `hivelink models list` (with state column), `hivelink models remove <model>` (with confirmation prompt, `--yes` to skip)
- [x] Model card shows download button directly in Models tab for any known-but-not-cached model — click opens the Pull modal pre-filled with that model name

---

## ✅ v0.3.5 — Model sync across cluster

> Goal: pull a model once on the fastest/most-connected node, automatically sync it to every other node that needs it for a cluster-split run — instead of manually running `ollama pull` on each machine separately.
>
> **Scope note:** full-copy + per-node layer loading approach, not byte-level GGUF slicing. Each node ends up with a complete local copy; the win is automating the copy + verifying it's ready. True partial-file streaming is tracked separately as a research item (v0.3.6).

- [x] Node-to-node file transfer — chunked HTTP via `GET /api/sync/blob/{digest}` streamed from whichever node has it cached
- [x] Resume-on-failure — transfer resumes from last-good chunk, not from zero
- [x] Checksum verification — SHA256 verified against source node before marking usable
- [x] "Sync to cluster" button in dashboard — one-click sync with live per-node progress bars
- [x] CLI equivalent: `hivelink sync <model>`

---

## 🔬 v0.3.6 — True partial-file model streaming (research item, not committed)

> Goal (aspirational): each node downloads *only* the bytes for its assigned layers, not the full model file — real bandwidth and disk savings for very large models split across many nodes.

- [ ] Investigate GGUF tensor index format well enough to determine whether a valid partial file (containing only a layer subset + a rewritten index) can be constructed reliably
- [ ] Prototype against one well-understood architecture (e.g. Llama family) before generalizing
- [ ] Decision point after prototype: pursue as a real feature, or formally close out as "not pursuing, full-copy sync is sufficient" with reasoning recorded here
- [ ] If pursued: needs its own dedicated milestone — this entry is scoping/research only, not a commitment

---

## ✅ v0.4 — Sharding controls + instances

> Goal: match EXO's instance/sharding UI, give users real control over how models run across the cluster.

- [x] Sharding mode selector — Pipeline (current) vs Tensor (see research spike below) vs Auto
- [x] Auto sharding — detects connection type (Ethernet / WiFi / Thunderbolt) per node, picks the best safe mode automatically; shown as "Auto → Pipeline (Ethernet)" in the plan view
- [x] Connection type badges on node cards — ⬡ ETH / ⌾ WiFi / ⚡ TB per node in the cluster view
- [x] Minimum nodes setting — only launch inference if at least N nodes are available; clear error if not met
- [x] Instance management — launch, monitor, and kill model instances; models pinned in GPU memory via Ollama keep_alive
- [x] Instance panel in dashboard — "Running instances" card on Models page; per-instance status badge, uptime clock, Kill button; Launch button on every cached model card; real-time updates over WebSocket

---

## 🔬 Tensor parallelism (research spike, not committed — moved out of v0.4)

> **Why this isn't in v0.4:** Tensor parallelism splits individual matrix operations across nodes
> simultaneously, requiring an all-reduce communication step between every layer. Ollama has no
> concept of splitting a single operation across network-connected machines — `--tensor-split` in
> llama.cpp only works across GPUs on the *same* machine. Real cross-network tensor parallelism
> (as used by vLLM in data center setups) requires writing custom all-reduce coordination over
> TCP/NCCL — a multi-month research project that would mean replacing Ollama entirely.
>
> The Tensor button is kept in the UI with an honest explanation rather than being removed,
> so users understand why it's not available. This spike tracks what it would take to ever change that.

- [ ] Investigate whether llama.cpp's `--rpc` server mode (added in late 2024) supports true tensor splitting across network nodes — it may provide a path that doesn't require writing all-reduce from scratch
- [ ] Prototype a single-layer boundary handoff between two llama.cpp RPC nodes — measure latency overhead per all-reduce step on a home GbE LAN
- [ ] Decision point after prototype: pursue as a real feature (would need its own milestone, v1.x territory), or formally close out as "not viable on home LAN hardware" with benchmarks recorded here

---

## 🔨 v0.5 — Native installers

> Goal: one-click install on every platform, no terminal required for end users.

- [ ] **Windows** — `.exe` installer (PyInstaller + NSIS), system tray icon, WebView2 app window
- [ ] **macOS** — `.dmg` with `.app` bundle (PyInstaller + Swift menu bar wrapper), WKWebView window, LaunchAgent auto-start
- [ ] **Linux** — AppImage (any distro, no install), `.deb` for Ubuntu/Debian, `.rpm` for Fedora
- [ ] Auto-updater — checks GitHub releases, downloads and applies updates in background
- [ ] First-run setup wizard — detects hardware, installs llama.cpp, downloads a starter model

---

## ✅ v0.6 — Vision + file uploads

> Goal: multimodal support in the chat tab.

- [x] Auto-detect vision-capable models (llava, llama3.2-vision, Qwen2-VL, LLaVA, MiniCPM-V, Moondream, etc.) — 👁 badge in model dropdown, upload button only shown for vision models
- [x] File upload button in chat — paperclip icon, only visible when selected model supports vision
- [x] Image preview in chat bubbles — thumbnail shown above user message
- [x] PDF upload — PDF.js extracts text client-side, injected as context in the message
- [x] Drag-and-drop into chat messages area — images and PDFs both supported

---

## 🔨 v0.7 — Cross-network clusters

> Goal: connect nodes that aren't on the same LAN.

- [ ] Tailscale integration — `hivelink join --tailscale` auto-configures discovery over Tailscale subnet
- [ ] `hivelink peer add <ip>:<port>` CLI subcommand — static peer management via CLI (underlying `HIVELINK_PEERS` env var + `--peer` flag already works today; this wraps it in a proper subcommand with persistent config)
- [ ] Encrypted transport — TLS between nodes for cross-network inference
- [ ] Latency-aware scheduling — deprioritize high-latency nodes for layer assignments

---

## 🔨 v0.8 — Multi-engine support (per-node engine choice)

> Goal: let each node run whichever engine the user prefers — Ollama, MLX, vLLM, llama-server — and pick which one to use per-model from the dashboard. Cross-node *splitting* stays on llama.cpp/GGUF (the only format that works the same across CUDA/Metal/ROCm); this is about choosing a full standalone model+engine per node, not cross-engine layer splitting.

- [ ] Model dropdown shows engine alongside model name — e.g. "llama3.2 (MLX · Mac mini)" vs "llama3.2 (Ollama · Windows)"
- [ ] Dashboard lets you pick which node/engine handles a given chat request when not using cluster-split mode
- [ ] Document the constraint clearly in UI: cross-node layer splitting requires llama.cpp/GGUF on all participating nodes; MLX/vLLM nodes serve standalone (non-split) models only, selectable individually
- [ ] (Stretch, no committed timeline) Investigate cross-engine pipeline splitting — different tensor formats and activation protocols make this a hard problem, likely its own research spike rather than a quick feature

---

## ✅ v0.9 — Tool calling + Skills

> Goal: let local models use custom skill contexts, and pass tools through to capable models.

- [x] OpenAI-compatible tool calling — `ChatRequest` now accepts `tools` and `tool_choice`, passed straight through to Ollama for capable models (Qwen2.5, Llama3.1+, Mistral)
- [x] Skills system — create, edit, delete, import from URL (JSON or Vercel-style markdown with YAML frontmatter); stored in `~/.hivelink/skills.json`
- [x] Skill elicitation UI — modal collects required input values before activating a skill; values interpolated into system prompt via `{variable}` syntax
- [x] Tools tab in dashboard — skills library with browse/create/import/edit/delete; tool-capable model reference
- [x] Active skill selector in Chat tab header — skill system prompt injected automatically on every message
- [x] Tool-capable model badges in Tools tab

---

## 🔨 v0.9.5 — MCP bridge + Artifact panel + Web search

> Goal: connect to MCP servers, give the model web search, render generated content in a side panel.

- [ ] MCP bridge — `POST /api/mcp/connect`, `GET /api/mcp/tools`, `POST /api/mcp/call`; manages HTTP/SSE connections to MCP servers
- [ ] MCP panel in Tools tab — add/remove MCP servers, see connection status, browse available tools
- [ ] Auto-inject connected MCP tools into chat requests when a capable model is selected
- [ ] Built-in web search tool — DuckDuckGo or Brave Search API, usable by any tool-capable model
- [ ] Tool-call indicator in chat — "Calling web_search…" animation during tool execution
- [ ] Artifact/preview side panel — renders generated markdown, code, and documents outside the chat bubble (like Claude's artifact panel)

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

## 🔨 v0.12 — Image/video generation routing

> Goal: route image and video generation requests to whichever cluster node has a generation backend (ComfyUI, Automatic1111, Diffusers) loaded. **Not** distributed splitting like LLM inference — diffusion/DiT model architectures don't have the same clean sequential layer boundary transformers do, so the value here is node *routing*, not layer-splitting a single generation across machines.

- [ ] New `/v1/images/generations` endpoint (OpenAI-spec compatible) — HiveLink routes the request to the best-fit node, not split across nodes
- [ ] Generation backend detection — extend the existing engine-detection pattern (`stats.py`) to recognize ComfyUI / Automatic1111 / Diffusers servers running on a node, similar to how Ollama/MLX/vLLM are detected today
- [ ] Per-node VRAM feasibility check for image/video — different math from `scheduler.py`'s LLM model fit, since diffusion VRAM use spikes during denoising steps rather than staying steady; video models (CogVideoX, Mochi, etc.) multiply that further by frame count
- [ ] New "Media" tab in dashboard — shows generated images/video, which node produced them, generation time
- [ ] Node selection picks single best-fit node for the job — closer to v0.8's multi-engine routing model than to true distributed LLM inference
- [ ] Video generation comes after image generation is solid — much higher VRAM and longer generation time, scope separately once image routing is proven out
- [ ] No standard OpenAI-spec endpoint exists yet for video generation — will likely need a HiveLink-specific endpoint until one is established industry-wide

---

## 💡 Future ideas (unscheduled)

- **watchOS / Android companion** — cluster status glanceable on wrist or phone
- **Web-hosted dashboard** — shareable cluster status page (read-only, no inference)
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
