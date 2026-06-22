"""
hivelink.discovery
UDP broadcast peer discovery. Nodes announce themselves every 3 seconds.
No configuration needed — just run on the same LAN.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore
from dataclasses import dataclass, field, asdict
from typing import Callable

from .hardware import HardwareProfile

logger = logging.getLogger(__name__)

DISCOVERY_PORT    = 47731
ANNOUNCE_INTERVAL = 3.0
PEER_TIMEOUT      = 12.0
BROADCAST_ADDR    = "255.255.255.255"

# Virtual adapter prefixes to skip when picking the "real" LAN IP — these are
# typically VMware/VirtualBox/Hyper-V/Docker virtual networks, not the actual
# home/office LAN. UDP broadcast on these never reaches other physical machines.
_VIRTUAL_ADAPTER_PREFIXES = (
    "192.168.56.",   # VirtualBox default host-only network
    "192.168.99.",   # Docker Toolbox
    "172.17.",        # Docker default bridge
    "172.18.", "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
    "172.30.", "172.31.",  # Docker/Hyper-V ranges
    "169.254.",        # Link-local (no real connectivity)
)


def _get_real_lan_ips() -> list[str]:
    """
    Return all non-virtual local IPv4 addresses on this machine, best guess first.
    Used to broadcast discovery packets on the actual LAN instead of a VM/Docker
    virtual adapter that happens to be the OS's default route.
    """
    ips: list[str] = []
    try:
        import psutil
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip = addr.address
                    if ip.startswith("127.") or any(ip.startswith(p) for p in _VIRTUAL_ADAPTER_PREFIXES):
                        continue
                    ips.append(ip)
    except ImportError:
        pass

    # Fallback: the classic "connect to 8.8.8.8 to find our outbound IP" trick —
    # doesn't actually send packets, just asks the OS routing table.
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if not any(ip.startswith(p) for p in _VIRTUAL_ADAPTER_PREFIXES):
                ips.append(ip)
        except Exception:
            pass

    return ips


def _broadcast_addr_for_ip(ip: str) -> str:
    """Compute the /24 broadcast address for a given IP, e.g. 192.168.1.115 -> 192.168.1.255."""
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.255"
    return BROADCAST_ADDR


@dataclass
class PeerInfo:
    node_id: str
    hostname: str
    api_port: int
    hardware: dict
    last_seen: float = field(default_factory=time.monotonic)
    is_self: bool = False

    @property
    def api_url(self) -> str:
        return f"http://{self.hostname}:{self.api_port}"

    @property
    def memory_score_mb(self) -> int:
        vram = self.hardware.get("total_vram_mb", 0)
        ram  = self.hardware.get("ram_mb", 0)
        return vram if vram > 0 else ram

    @property
    def tflops(self) -> float:
        return self.hardware.get("total_fp16_tflops", 1.0)

    @property
    def connection_type(self) -> str:
        """Connection type reported by this peer's hardware profile."""
        return self.hardware.get("connection_type", "unknown")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("last_seen", None)
        return d


class DiscoveryService:
    def __init__(
        self,
        node_id: str,
        api_port: int,
        hardware: HardwareProfile,
        on_peer_change: Callable[[dict[str, PeerInfo]], None] | None = None,
        static_peers: list[str] | None = None,
    ):
        self.node_id        = node_id
        self.api_port       = api_port
        self.hardware       = hardware
        self.on_peer_change = on_peer_change
        self.static_peers   = static_peers or []   # list of "ip" or "ip:port"
        self.peers: dict[str, PeerInfo] = {}
        self._hostname = socket.gethostname()
        self._running  = False

    def _build_announce(self) -> bytes:
        return json.dumps({
            "node_id":  self.node_id,
            "hostname": self._hostname,
            "api_port": self.api_port,
            "hardware": self.hardware.to_dict(),
            "ts":       time.time(),
        }).encode()

    async def _broadcaster(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)
        loop = asyncio.get_running_loop()

        # Compute real LAN broadcast targets once at startup. Broadcasting on the
        # actual subnet (e.g. 192.168.1.255) instead of only the global address
        # (255.255.255.255) avoids announces silently going out on a VM/Docker
        # virtual adapter that happens to be the OS's chosen default route.
        real_ips = _get_real_lan_ips()
        targets  = {BROADCAST_ADDR}
        for ip in real_ips:
            targets.add(_broadcast_addr_for_ip(ip))
        if real_ips:
            logger.info("Broadcasting discovery on real LAN IP(s): %s -> targets %s",
                        real_ips, sorted(targets))
        else:
            logger.warning("Could not detect a real LAN IP — falling back to global "
                           "broadcast only. If discovery doesn't find peers across "
                           "machines, use --peer <ip> as a manual fallback.")

        while self._running:
            try:
                packet = self._build_announce()
                for target in targets:
                    try:
                        await loop.sock_sendto(sock, packet, (target, DISCOVERY_PORT))
                    except Exception as e:
                        logger.debug("Broadcast to %s failed: %s", target, e)
            except Exception as e:
                logger.debug("Broadcast error: %s", e)
            await asyncio.sleep(ANNOUNCE_INTERVAL)
        sock.close()

    async def _listener(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # Windows doesn't have SO_REUSEPORT
        sock.bind(("", DISCOVERY_PORT))
        sock.setblocking(False)
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                data, addr = await loop.sock_recvfrom(sock, 4096)
                await self._handle_announce(data, addr[0])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Listener error: %s", e)
        sock.close()

    async def _reaper(self) -> None:
        while self._running:
            await asyncio.sleep(PEER_TIMEOUT / 2)
            now   = time.monotonic()
            stale = [
                nid for nid, p in self.peers.items()
                if not p.is_self and now - p.last_seen > PEER_TIMEOUT
            ]
            if stale:
                for nid in stale:
                    logger.info("Peer timed out: %s", nid)
                    del self.peers[nid]
                self._notify()

    async def _handle_announce(self, data: bytes, src_ip: str) -> None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return

        node_id = payload.get("node_id")
        if not node_id:
            return

        is_self  = node_id == self.node_id
        existing = self.peers.get(node_id)

        peer = PeerInfo(
            node_id  = node_id,
            hostname = src_ip,
            api_port = payload.get("api_port", 47730),
            hardware = payload.get("hardware", {}),
            last_seen= time.monotonic(),
            is_self  = is_self,
        )

        self.peers[node_id] = peer
        if not existing:
            logger.info("New peer: %s @ %s (%s)",
                        node_id[:8], src_ip,
                        peer.hardware.get("primary_backend", "?"))
            self._notify()

    def _notify(self) -> None:
        if self.on_peer_change:
            self.on_peer_change(self.peers)

    async def _static_peer_poller(self) -> None:
        """Poll static peers by IP every ANNOUNCE_INTERVAL seconds."""
        if not _httpx or not self.static_peers:
            return
        while self._running:
            for addr in self.static_peers:
                if ":" in addr:
                    ip, port = addr.rsplit(":", 1)
                    port = int(port)
                else:
                    ip, port = addr, 47730
                try:
                    async with _httpx.AsyncClient(timeout=2) as client:
                        resp = await client.get(f"http://{ip}:{port}/api/cluster")
                        if resp.status_code == 200:
                            data = resp.json()
                            for peer_data in data.get("peers", []):
                                node_id = peer_data.get("node_id")
                                if not node_id or node_id == self.node_id:
                                    continue
                                hw = peer_data.get("hardware", {})
                                existing = self.peers.get(node_id)
                                self.peers[node_id] = PeerInfo(
                                    node_id  = node_id,
                                    hostname = ip,
                                    api_port = peer_data.get("api_port", port),
                                    hardware = hw,
                                    last_seen= time.monotonic(),
                                    is_self  = False,
                                )
                                if not existing:
                                    logger.info("Static peer connected: %s @ %s", node_id[:8], ip)
                                    self._notify()
                                else:
                                    # refresh last_seen — reaper won't kill it
                                    self.peers[node_id].last_seen = time.monotonic()
                except Exception as e:
                    logger.debug("Static peer poll error %s: %s", ip, e)
            await asyncio.sleep(ANNOUNCE_INTERVAL)

    async def start(self) -> None:
        self._running = True
        self.peers[self.node_id] = PeerInfo(
            node_id  = self.node_id,
            hostname = self._hostname,
            api_port = self.api_port,
            hardware = self.hardware.to_dict(),
            is_self  = True,
        )
        logger.info("Discovery started — node_id=%s port=%d", self.node_id[:8], DISCOVERY_PORT)
        tasks = [self._broadcaster(), self._listener(), self._reaper()]
        if self.static_peers:
            logger.info("Static peers configured: %s", self.static_peers)
            tasks.append(self._static_peer_poller())
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False

    def active_peers(self) -> list[PeerInfo]:
        return sorted(self.peers.values(), key=lambda p: p.tflops, reverse=True)

    def remote_peers(self) -> list[PeerInfo]:
        return [p for p in self.active_peers() if not p.is_self]
