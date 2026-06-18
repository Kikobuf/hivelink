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


@app.command()
def start(
    port:   int      = typer.Option(47730,    "--port",   "-p", help="API server port"),
    host:   str      = typer.Option("0.0.0.0","--host",         help="Bind host"),
    reload: bool     = typer.Option(False,    "--reload",        help="Dev mode auto-reload"),
    peer:   list[str]= typer.Option([],       "--peer",   "-P",  help="Static peer IP[:port] — use when UDP discovery can't cross subnets. Can repeat: --peer 192.168.1.112 --peer 10.0.0.5"),
):
    """Start the HiveLink node. Joins the cluster via UDP discovery or static peers."""
    os.environ["HIVELINK_PORT"] = str(port)
    if peer:
        os.environ["HIVELINK_PEERS"] = ",".join(peer)
        console.print(f"[dim]Static peers:[/] {', '.join(peer)}")
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


@app.command()
def models(port: int = typer.Option(47730, "--port", "-p")):
    """List models this cluster can run."""
    import httpx
    try:
        data = httpx.get(f"http://localhost:{port}/api/models", timeout=3).json()
    except Exception:
        rprint("[red]Cannot connect to HiveLink.[/]")
        raise typer.Exit(1)

    table = Table(title="Runnable Models")
    table.add_column("Model")
    table.add_column("Params",  justify="right")
    table.add_column("Quant")
    table.add_column("Size",    justify="right")

    for m in data["models"]:
        table.add_row(m["model_id"], f"{m['params_b']:.0f}B",
                      f"Q{m['quant_bits']}", f"{m['size_mb']:,} MB")
    console.print(table)


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


if __name__ == "__main__":
    app()
