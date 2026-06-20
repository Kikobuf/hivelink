"""
hivelink.cli
Commands: start, status, models, plan, hardware, install-llama
"""

from __future__ import annotations

import os
import platform
import urllib.request
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app     = typer.Typer(
    name="hivelink",
    help="Cross-platform distributed LLM inference — pool any hardware into one AI cluster",
    no_args_is_help=True,
)
console = Console()


def _ensure_ollama_running() -> None:
    """
    If Ollama is installed but not running, start it in the background so the
    user doesn't have to remember a separate `ollama serve` step every time.
    Silent no-op if Ollama isn't installed, or is already running, or fails to
    start for any reason — this is a convenience, never a hard requirement.
    """
    import shutil
    import subprocess
    import httpx

    # Already running? Nothing to do.
    try:
        httpx.get("http://127.0.0.1:11434/api/tags", timeout=1)
        return
    except Exception:
        pass

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return  # Ollama not installed — nothing to auto-start

    try:
        if platform.system().lower() == "windows":
            # CREATE_NO_WINDOW alone (not combined with DETACHED_PROCESS — that
            # combination causes a repeated console flash-open/close loop on
            # some Windows builds as the process tries to attach/detach a console).
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                [ollama_bin, "serve"],
                creationflags=CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
        else:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,  # detach from this process group on Mac/Linux
            )
        console.print("[dim]Ollama wasn't running — started it in the background.[/]")
    except Exception as e:
        console.print(f"[dim]Couldn't auto-start Ollama ({e}) — start it manually with `ollama serve` if needed.[/]")


@app.command()
def start(
    port:        int      = typer.Option(47730,    "--port",   "-p", help="API server port"),
    host:        str      = typer.Option("0.0.0.0","--host",         help="Bind host"),
    reload:      bool     = typer.Option(False,    "--reload",        help="Dev mode auto-reload"),
    peer:        list[str]= typer.Option([],       "--peer",   "-P",  help="Static peer IP[:port] — use when UDP discovery can't cross subnets. Can repeat: --peer 192.168.1.112 --peer 10.0.0.5"),
    auto_ollama: bool     = typer.Option(True,      "--auto-ollama/--no-auto-ollama", help="Automatically start Ollama in the background if it's installed but not running"),
):
    """Start the HiveLink node. Joins the cluster via UDP discovery or static peers."""
    os.environ["HIVELINK_PORT"] = str(port)
    if peer:
        os.environ["HIVELINK_PEERS"] = ",".join(peer)
        console.print(f"[dim]Static peers:[/] {', '.join(peer)}")

    if auto_ollama:
        _ensure_ollama_running()

    console.print(f"\n[bold green]HiveLink[/] starting on [cyan]http://{host}:{port}[/]")
    console.print(f"Dashboard: [cyan]http://localhost:{port}[/]\n")
    uvicorn.run(
        "hivelink.server:app",
        host=host, port=port, reload=reload, log_level="info",
    )


@app.command()
def status(port: int = typer.Option(47730, "--port", "-p")):
    """Show current cluster status."""
    import httpx
    try:
        resp = httpx.get(f"http://localhost:{port}/api/cluster", timeout=3)
        data = resp.json()
    except Exception:
        rprint("[red]Cannot connect to HiveLink. Is it running? Try: hivelink start[/]")
        raise typer.Exit(1)

    peers = data["peers"]
    table = Table(title=f"HiveLink Cluster — {len(peers)} node(s)")
    table.add_column("Node",    style="cyan")
    table.add_column("OS")
    table.add_column("Backend", style="green")
    table.add_column("VRAM / RAM")
    table.add_column("TFLOPS", justify="right")
    table.add_column("Self?")

    for p in peers:
        hw  = p.get("hardware", {})
        mem = hw.get("total_vram_mb", 0) or hw.get("ram_mb", 0)
        table.add_row(
            p["node_id"][:12],
            hw.get("node_os", "?"),
            hw.get("primary_backend", "?"),
            f"{mem:,} MB",
            f"{hw.get('total_fp16_tflops', 0):.1f}",
            "✓" if p.get("is_self") else "",
        )

    console.print(table)
    console.print(f"\nTotal memory: [bold]{data['total_vram_mb']:,} MB[/]  |  "
                  f"Total TFLOPS: [bold]{data['total_tflops']}[/]")


models_app = typer.Typer(help="List, pull, and remove models")
app.add_typer(models_app, name="models")


@models_app.command(name="list")
def models_list(port: int = typer.Option(47730, "--port", "-p")):
    """List models this cluster can run, and their state (live / cached / pullable)."""
    import httpx
    try:
        data = httpx.get(f"http://localhost:{port}/api/models", timeout=3).json()
    except Exception:
        rprint("[red]Cannot connect to HiveLink.[/]")
        raise typer.Exit(1)

    table = Table(title="Models")
    table.add_column("Model")
    table.add_column("State")
    table.add_column("Params",  justify="right")
    table.add_column("Quant")
    table.add_column("Size",    justify="right")

    for m in data["models"]:
        if m.get("live"):
            state = "[green]● live[/]"
        elif m.get("cached"):
            state = "[cyan]⬇ cached[/]"
        else:
            state = "[dim]pullable[/]"
        table.add_row(m["model_id"], state, f"{m['params_b']:.1f}B",
                      f"Q{m['quant_bits']}", f"{m['size_mb']:,} MB")
    console.print(table)


@models_app.command(name="remove")
def models_remove(
    model_id: str = typer.Argument(..., help="Model to remove, e.g. llama3.2, qwen2.5:7b"),
    port:     int = typer.Option(47730, "--port", "-p"),
    yes:      bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Remove a cached model from this node to free up disk space."""
    import httpx

    if not yes:
        confirm = typer.confirm(f"Remove '{model_id}' from this node?")
        if not confirm:
            rprint("[dim]Cancelled.[/]")
            raise typer.Exit(0)

    try:
        resp = httpx.delete(f"http://localhost:{port}/api/models/{model_id}", timeout=15)
        if resp.status_code == 200:
            console.print(f"[green]✓[/] Removed [cyan]{model_id}[/]")
        elif resp.status_code == 503:
            rprint("[red]Ollama is not running on this node.[/]")
            raise typer.Exit(1)
        else:
            rprint(f"[red]Failed to remove model — server returned {resp.status_code}[/]")
            raise typer.Exit(1)
    except httpx.ConnectError:
        rprint("[red]Cannot connect to HiveLink. Is it running?[/]")
        raise typer.Exit(1)


# Backwards-compatible alias: `hivelink models` with no subcommand behaves like `models list`
@app.command(name="models", hidden=True)
def models_legacy(port: int = typer.Option(47730, "--port", "-p")):
    """(Deprecated alias — use `hivelink models list`)"""
    models_list(port=port)


@app.command()
def plan(
    model_id:   str = typer.Argument(..., help="Model ID e.g. llama3-70b"),
    quant_bits: int = typer.Option(4, "--quant", "-q"),
    port:       int = typer.Option(47730, "--port", "-p"),
):
    """Show layer distribution plan for a model across the cluster."""
    import httpx
    try:
        resp = httpx.get(f"http://localhost:{port}/api/plan/{model_id}",
                         params={"quant_bits": quant_bits}, timeout=3)
        if resp.status_code == 422:
            rprint(f"[red]Cluster cannot run {model_id} at Q{quant_bits} — need more memory.[/]")
            raise typer.Exit(1)
        data = resp.json()
    except Exception as e:
        rprint(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    table = Table(title=f"{model_id} Q{quant_bits} — {data['total_layers']} layers, "
                        f"{data['node_count']} node(s)")
    table.add_column("Node")
    table.add_column("Backend")
    table.add_column("Layers",      justify="right")
    table.add_column("Layer range")
    table.add_column("~VRAM",       justify="right")

    for a in data["assignments"]:
        table.add_row(a["node_id"][:12], a["backend"],
                      str(a["layer_count"]),
                      f"{a['layer_start']}–{a['layer_end']}",
                      f"{a['vram_mb']:,} MB")
    console.print(table)


@app.command()
def hardware():
    """Detect and display this machine's hardware."""
    from .hardware import detect
    p = detect()
    console.print(f"\n[bold]Hardware Profile[/]")
    console.print(f"  OS:      {p.node_os}")
    console.print(f"  CPU:     {p.cpu_name} ({p.cpu_cores} cores)")
    console.print(f"  RAM:     {p.ram_mb:,} MB")
    console.print(f"  Backend: [green]{p.primary_backend}[/]")
    if p.gpus:
        console.print("\n  [bold]GPUs:[/]")
        for g in p.gpus:
            console.print(f"    [{g.index}] {g.name}")
            console.print(f"        VRAM: {g.vram_mb:,} MB | ~{g.fp16_tflops:.1f} FP16 TFLOPS")
    else:
        console.print("  [yellow]No GPU detected — CPU inference only[/]")


@app.command(name="install-llama")
def install_llama():
    """Download the llama.cpp server binary for this platform."""
    os_name = platform.system().lower()
    machine = platform.machine().lower()

    RELEASES = {
        ("linux",   "x86_64"):  "llama-server-linux-x64",
        ("linux",   "aarch64"): "llama-server-linux-arm64",
        ("darwin",  "arm64"):   "llama-server-macos-arm64",
        ("darwin",  "x86_64"):  "llama-server-macos-x64",
        ("windows", "amd64"):   "llama-server-win-x64.exe",
        ("windows", "x86_64"):  "llama-server-win-x64.exe",
    }

    key = (os_name, machine)
    binary_name = RELEASES.get(key)
    if not binary_name:
        rprint(f"[red]No prebuilt binary for {os_name}/{machine}. "
               "Build from source: https://github.com/ggml-org/llama.cpp[/]")
        raise typer.Exit(1)

    url  = f"https://github.com/ggml-org/llama.cpp/releases/latest/download/{binary_name}"
    dest = Path.home() / ".hivelink" / "bin" / binary_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"Downloading [cyan]{binary_name}[/]...")
    urllib.request.urlretrieve(url, dest)
    if os_name != "windows":
        dest.chmod(0o755)

    console.print(f"[green]Installed to {dest}[/]")
    console.print(f"\nStart llama-server with:")
    console.print(f"  [cyan]{dest} --model path/to/model.gguf --port 8080[/]")


@app.command()
def pull(
    model_id: str = typer.Argument(..., help="Model to pull, e.g. llama3.2, qwen2.5:7b, mistral"),
    port:     int = typer.Option(47730, "--port", "-p", help="HiveLink API port (used to detect local engines)"),
):
    """
    Pull a model so it's available to the cluster.

    Detects which inference engine is running locally and uses the right
    pull mechanism:
      - Ollama running    -> `ollama pull <model>` (streams progress)
      - No engine running -> prints setup instructions

    Once pulled, the model shows up automatically in `hivelink models`
    and the dashboard's Models tab on every node running this engine.
    """
    import shutil
    import subprocess

    ollama_bin = shutil.which("ollama")

    if ollama_bin:
        console.print(f"[bold]Pulling[/] [cyan]{model_id}[/] via Ollama...\n")
        try:
            # ollama pull streams its own progress bar to stdout — just pipe it through
            result = subprocess.run([ollama_bin, "pull", model_id])
            if result.returncode == 0:
                console.print(f"\n[green]✓[/] {model_id} pulled successfully.")
                console.print(f"It will appear in [cyan]hivelink models[/] and the dashboard "
                               f"once Ollama has it loaded.")
            else:
                rprint(f"[red]ollama pull exited with code {result.returncode}[/]")
                raise typer.Exit(1)
        except FileNotFoundError:
            rprint("[red]Ollama binary found on PATH but failed to execute.[/]")
            raise typer.Exit(1)
        return

    # No engine detected — check if HiveLink itself is running to give a more specific hint
    import httpx
    try:
        httpx.get(f"http://localhost:{port}/health", timeout=2)
        hivelink_running = True
    except Exception:
        hivelink_running = False

    rprint("[yellow]No inference engine found on this machine.[/]\n")
    rprint("HiveLink doesn't bundle model downloading itself — it routes to "
           "whichever engine (Ollama, llama-server, MLX, vLLM) you have installed.\n")
    rprint("[bold]Easiest option — install Ollama:[/]")
    if platform.system().lower() == "darwin":
        rprint("  [cyan]brew install ollama[/]  (or download from https://ollama.com)")
    elif platform.system().lower() == "windows":
        rprint("  Download from [cyan]https://ollama.com/download/windows[/]")
    else:
        rprint("  [cyan]curl -fsSL https://ollama.com/install.sh | sh[/]")
    rprint(f"\nThen run: [cyan]hivelink pull {model_id}[/] again\n")

    if hivelink_running:
        rprint("[dim]HiveLink is running but found no engine — once Ollama is installed "
               "and this model is pulled, it'll be auto-detected within a few seconds.[/]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
