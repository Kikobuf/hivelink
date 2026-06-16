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
from dataclasses import dataclass, field, asdict
from typing import Callable

from .hardware import HardwareProfile

logger = logging.getLogger(__name__)

DISCOVERY_PORT    = 47731
ANNOUNCE_INTERVAL = 3.0
PEER_TIMEOUT      = 12.0
BROADCAST_ADDR    = "255.255.255.255"


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
    ):
        self.node_id       = node_id
        self.api_port      = api_port
        self.hardware      = hardware
        self.on_peer_change = on_peer_change
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
        while self._running:
            try:
                packet = self._build_announce()
                await loop.sock_sendto(sock, packet, (BROADCAST_ADDR, DISCOVERY_PORT))
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
        await asyncio.gather(
            self._broadcaster(),
            self._listener(),
            self._reaper(),
        )

    async def stop(self) -> None:
        self._running = False

    def active_peers(self) -> list[PeerInfo]:
        return sorted(self.peers.values(), key=lambda p: p.tflops, reverse=True)

    def remote_peers(self) -> list[PeerInfo]:
        return [p for p in self.active_peers() if not p.is_self]
