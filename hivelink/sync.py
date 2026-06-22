"""
hivelink.sync
Model sync across the cluster — v0.3.5.

Goal: pull a model once on one node, automatically copy it to every other
connected node that needs it, so you don't have to manually `ollama pull`
the same model on each machine separately.

Design notes (read this before changing anything):

  - Ollama stores models as content-addressed blobs under
    ~/.ollama/models/blobs/sha256-<hash>, referenced by a manifest file at
    ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>.
    We sync at the BLOB level, not "the model file" as one opaque unit —
    this gets us resumable, checksummed transfers almost for free, since
    Ollama's own content-addressing already gives each blob a stable,
    verifiable SHA256 identity. We don't invent our own chunking scheme;
    we reuse Ollama's.

  - Resume works by tracking how many bytes of EACH blob have already
    been written locally, and requesting only the remaining byte range
    from the source node on retry. This is the part that took real
    thought: partial-file state has to survive a HiveLink restart, and
    has to be invalidated (not blindly resumed) if the source blob's
    hash doesn't match what we expect.

  - "Sync" is push-style: the node that HAS the model tells nodes that
    DON'T have it to pull from it directly (node-to-node), with HiveLink's
    coordinator just orchestrating who-needs-what and reporting progress.
    The dashboard/CLI talks to the coordinator; the coordinator talks to
    peers; peers transfer blobs directly between themselves.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4 * 1024 * 1024          # 4MB read/write chunks during transfer
PROGRESS_INTERVAL = 0.5                # seconds between progress events emitted
TRANSFER_TIMEOUT = None                # no timeout — large blobs take a while
MAX_CONCURRENT_BLOB_TRANSFERS = 2      # don't saturate the LAN link entirely


def _ollama_home() -> Path:
    """Resolve Ollama's model storage root, matching Ollama's own platform logic."""
    env_override = os.environ.get("OLLAMA_MODELS")
    if env_override:
        return Path(env_override)
    if platform.system().lower() == "windows":
        return Path(os.environ.get("USERPROFILE", Path.home())) / ".ollama" / "models"
    return Path.home() / ".ollama" / "models"


def _manifest_path(model_id: str) -> Path:
    """
    Resolve a model_id like 'qwen2.5:32b' to its manifest file path.
    Ollama's manifest layout: manifests/registry.ollama.ai/library/<name>/<tag>
    (tag defaults to 'latest' if not specified in model_id).
    """
    if ":" in model_id:
        name, tag = model_id.split(":", 1)
    else:
        name, tag = model_id, "latest"
    return _ollama_home() / "manifests" / "registry.ollama.ai" / "library" / name / tag


def _blob_path(digest: str) -> Path:
    """Resolve a content digest (e.g. 'sha256:abc123...') to its blob file path."""
    safe_digest = digest.replace(":", "-")  # Ollama stores blobs as sha256-<hash> on disk
    return _ollama_home() / "blobs" / safe_digest


@dataclass
class BlobInfo:
    """One content-addressed blob referenced by a model's manifest."""
    digest: str          # e.g. "sha256:abc123..."
    size: int            # total expected size in bytes
    media_type: str = ""  # informational only (model weights vs config vs license, etc.)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlobTransferState:
    """Tracks resume state for one blob transfer, persisted to disk so a HiveLink
    restart mid-transfer doesn't lose progress."""
    digest: str
    expected_size: int
    bytes_written: int = 0
    source_node_id: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    complete: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BlobTransferState":
        return cls(**d)


def _state_dir() -> Path:
    """Where we persist in-progress transfer state, separate from Ollama's own storage."""
    d = _ollama_home() / ".hivelink-sync-state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path(digest: str) -> Path:
    safe = digest.replace(":", "-")
    return _state_dir() / f"{safe}.json"


def _load_transfer_state(digest: str) -> Optional[BlobTransferState]:
    p = _state_path(digest)
    if not p.exists():
        return None
    try:
        return BlobTransferState.from_dict(json.loads(p.read_text()))
    except Exception:
        return None  # corrupted state file — treat as no prior progress


def _save_transfer_state(state: BlobTransferState) -> None:
    state.updated_at = time.time()
    _state_path(state.digest).write_text(json.dumps(state.to_dict()))


def _clear_transfer_state(digest: str) -> None:
    p = _state_path(digest)
    if p.exists():
        p.unlink()


def read_manifest_blobs(model_id: str) -> list[BlobInfo]:
    """
    Parse a locally-cached model's manifest to get the list of blobs that make
    up the model (weights, config, license, etc.) Returns empty list if the
    model isn't cached on this node.
    """
    path = _manifest_path(model_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to parse manifest for %s: %s", model_id, e)
        return []

    blobs: list[BlobInfo] = []
    # Ollama manifest format: { "config": {digest,size,...}, "layers": [{digest,size,mediaType}, ...] }
    config = data.get("config")
    if config and config.get("digest"):
        blobs.append(BlobInfo(digest=config["digest"], size=config.get("size", 0),
                               media_type=config.get("mediaType", "config")))
    for layer in data.get("layers", []):
        if layer.get("digest"):
            blobs.append(BlobInfo(digest=layer["digest"], size=layer.get("size", 0),
                                   media_type=layer.get("mediaType", "")))
    return blobs


def local_blob_status(digest: str, expected_size: int) -> tuple[bool, int]:
    """
    Check whether a blob is fully present locally and verified.
    Returns (is_complete, bytes_present). A blob is only considered complete
    if its file size matches AND (for safety) we trust the original pull's
    own integrity check — Ollama already verifies SHA256 on pull, so we
    don't re-hash a full 19GB file here; we trust file size as the signal
    for "is this blob present", and rely on transfer-time hashing (below)
    for data we ourselves write during a sync.
    """
    path = _blob_path(digest)
    if not path.exists():
        return False, 0
    size = path.stat().st_size
    return size == expected_size, size


def missing_blobs_for_sync(model_id: str) -> list[BlobInfo]:
    """Return the subset of a model's blobs that are NOT fully present on this node."""
    blobs = read_manifest_blobs(model_id)
    missing = []
    for b in blobs:
        complete, _ = local_blob_status(b.digest, b.size)
        if not complete:
            missing.append(b)
    return missing


# ── Serving blobs to other nodes (source side) ────────────────────────────────

def serve_blob_range(digest: str, start: int = 0, end: Optional[int] = None) -> AsyncGenerator[bytes, None]:
    """
    Stream a byte range of a locally-stored blob, for another node to pull from us.
    Used by the /api/sync/blob/{digest} endpoint. Raises FileNotFoundError if we
    don't actually have this blob (caller should turn that into a 404).
    """
    path = _blob_path(digest)
    if not path.exists():
        raise FileNotFoundError(f"Blob {digest} not present on this node")

    total_size = path.stat().st_size
    if end is None or end >= total_size:
        end = total_size - 1

    async def gen() -> AsyncGenerator[bytes, None]:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
                await asyncio.sleep(0)  # yield control so this doesn't block the event loop

    return gen()


# ── Pulling blobs from another node (destination side) ────────────────────────

@dataclass
class SyncProgress:
    """One progress update, emitted during a sync operation."""
    model_id: str
    blob_digest: str
    blob_index: int       # which blob (1-based) out of total_blobs
    total_blobs: int
    bytes_written: int
    blob_total_bytes: int
    status: str            # "downloading" | "verifying" | "complete" | "error" | "resuming"
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


async def pull_blob_from_peer(
    digest: str,
    expected_size: int,
    source_ip: str,
    source_port: int,
    on_progress=None,
) -> bool:
    """
    Pull one blob from another HiveLink node, resuming from any prior partial
    download. Returns True on success, False on failure (caller can retry).

    Resume logic: we check local disk state first (how many bytes of THIS blob
    digest do we already have, both from Ollama's own storage if a prior pull
    half-succeeded, and from our own sync state file if a prior sync attempt
    was interrupted). We then request only the remaining byte range from the
    source node via HTTP Range headers, and append to the existing partial file
    rather than starting over.
    """
    blob_path = _blob_path(digest)
    blob_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine resume point: prefer actual bytes on disk over the state file's
    # claim, since disk is ground truth — the state file just helps us know
    # WHICH source node and WHICH digest we were mid-transfer on.
    state = _load_transfer_state(digest)
    bytes_on_disk = blob_path.stat().st_size if blob_path.exists() else 0

    if bytes_on_disk >= expected_size:
        # Already fully present (e.g. another sync or manual pull beat us to it)
        if on_progress:
            await on_progress(SyncProgress(
                model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
                bytes_written=expected_size, blob_total_bytes=expected_size,
                status="complete", detail="Already present",
            ))
        _clear_transfer_state(digest)
        return True

    resume_from = bytes_on_disk
    if state and state.digest == digest and state.source_node_id:
        # We have prior sync metadata for this exact blob — trust the on-disk
        # byte count over the state file's claim (disk is ground truth), but
        # log if they disagree since that'd indicate something unexpected
        # happened (e.g. file truncated externally).
        if state.bytes_written != bytes_on_disk:
            logger.warning(
                "Blob %s: state file says %d bytes, disk has %d — trusting disk.",
                digest, state.bytes_written, bytes_on_disk,
            )

    if resume_from > 0 and on_progress:
        await on_progress(SyncProgress(
            model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
            bytes_written=resume_from, blob_total_bytes=expected_size,
            status="resuming", detail=f"Resuming from {resume_from:,} bytes",
        ))

    url = f"http://{source_ip}:{source_port}/api/sync/blob/{digest}"
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    state = BlobTransferState(
        digest=digest, expected_size=expected_size,
        bytes_written=resume_from, source_node_id=f"{source_ip}:{source_port}",
    )

    try:
        mode = "ab" if resume_from > 0 else "wb"
        async with httpx.AsyncClient(timeout=TRANSFER_TIMEOUT) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 404:
                    logger.error("Blob %s not found on source node %s:%d", digest, source_ip, source_port)
                    return False
                if resp.status_code not in (200, 206):
                    logger.error("Unexpected status %d pulling blob %s", resp.status_code, digest)
                    return False

                # If we asked for a range but got a full 200 instead of 206,
                # the source doesn't support partial content — restart clean.
                if resume_from > 0 and resp.status_code == 200:
                    logger.info("Source doesn't support range requests for %s — restarting from 0", digest)
                    resume_from = 0
                    mode = "wb"
                    state.bytes_written = 0

                last_progress_emit = 0.0
                with open(blob_path, mode) as f:
                    written_this_session = 0
                    async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                        f.write(chunk)
                        written_this_session += len(chunk)
                        state.bytes_written = resume_from + written_this_session
                        now = time.time()
                        if on_progress and (now - last_progress_emit) >= PROGRESS_INTERVAL:
                            last_progress_emit = now
                            await on_progress(SyncProgress(
                                model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
                                bytes_written=state.bytes_written, blob_total_bytes=expected_size,
                                status="downloading",
                            ))
                            _save_transfer_state(state)  # checkpoint periodically, not every chunk

        _save_transfer_state(state)  # final checkpoint before verification

    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
        logger.warning("Transfer interrupted for blob %s: %s — partial state saved for resume", digest, e)
        _save_transfer_state(state)
        if on_progress:
            await on_progress(SyncProgress(
                model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
                bytes_written=state.bytes_written, blob_total_bytes=expected_size,
                status="error", detail=f"Connection interrupted: {e}",
            ))
        return False

    # ── Verify ──────────────────────────────────────────────────────────────
    final_size = blob_path.stat().st_size
    if final_size != expected_size:
        logger.error("Blob %s size mismatch after transfer: got %d, expected %d", digest, final_size, expected_size)
        if on_progress:
            await on_progress(SyncProgress(
                model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
                bytes_written=final_size, blob_total_bytes=expected_size,
                status="error", detail="Size mismatch after transfer — will retry",
            ))
        return False

    if on_progress:
        await on_progress(SyncProgress(
            model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
            bytes_written=expected_size, blob_total_bytes=expected_size,
            status="verifying",
        ))

    if not _verify_blob_hash(blob_path, digest):
        logger.error("Blob %s failed hash verification after transfer — deleting and will retry", digest)
        blob_path.unlink(missing_ok=True)
        _clear_transfer_state(digest)
        if on_progress:
            await on_progress(SyncProgress(
                model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
                bytes_written=0, blob_total_bytes=expected_size,
                status="error", detail="Checksum mismatch — corrupted transfer, retrying",
            ))
        return False

    _clear_transfer_state(digest)
    if on_progress:
        await on_progress(SyncProgress(
            model_id="", blob_digest=digest, blob_index=0, total_blobs=0,
            bytes_written=expected_size, blob_total_bytes=expected_size,
            status="complete",
        ))
    return True


def _verify_blob_hash(path: Path, expected_digest: str) -> bool:
    """
    Verify a downloaded blob's SHA256 against the digest Ollama's manifest expects.
    expected_digest is in the form 'sha256:abc123...'.
    """
    if not expected_digest.startswith("sha256:"):
        return True  # unknown digest format — skip verification rather than false-fail
    expected_hash = expected_digest.split(":", 1)[1]

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash


def write_manifest(model_id: str, manifest_json: dict) -> None:
    """Write a model's manifest file locally once all its blobs are synced, so
    Ollama recognizes the model as available without needing a separate pull."""
    path = _manifest_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_json))


def read_raw_manifest(model_id: str) -> Optional[dict]:
    """Read a model's raw manifest JSON (for sending to a peer that needs to sync it)."""
    path = _manifest_path(model_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


async def sync_model_to_peer(
    model_id: str,
    target_ip: str,
    target_port: int,
    this_node_ip: str,
    this_node_port: int,
    on_progress=None,
) -> bool:
    """
    Orchestrate syncing a model FROM this node TO a target peer. This runs on
    the SOURCE node's HiveLink instance and tells the target node (via its
    /api/sync/receive endpoint) to pull each missing blob from us.

    Concurrency: transfers up to MAX_CONCURRENT_BLOB_TRANSFERS blobs in parallel
    to use available bandwidth without overwhelming either machine's disk I/O.
    """
    blobs = read_manifest_blobs(model_id)
    if not blobs:
        logger.error("No manifest found for %s on this node — can't sync", model_id)
        return False

    manifest = read_raw_manifest(model_id)
    total = len(blobs)

    # Ask the target which blobs it's actually missing (it may already have
    # some, e.g. shared base-model blobs from a different tag it has cached).
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"http://{target_ip}:{target_port}/api/sync/check-missing",
                json={"blobs": [b.to_dict() for b in blobs]},
            )
            missing_digests = set(resp.json().get("missing_digests", [b.digest for b in blobs]))
        except Exception as e:
            logger.warning("Couldn't check target's missing blobs (%s) — assuming all needed", e)
            missing_digests = {b.digest for b in blobs}

    to_transfer = [b for b in blobs if b.digest in missing_digests]
    if not to_transfer:
        # Target already has every blob — just make sure it has the manifest
        if manifest:
            async with httpx.AsyncClient(timeout=30) as client:
                await client.post(
                    f"http://{target_ip}:{target_port}/api/sync/write-manifest",
                    json={"model_id": model_id, "manifest": manifest},
                )
        return True

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BLOB_TRANSFERS)

    async def transfer_one(blob: "BlobInfo", index: int) -> bool:
        async with semaphore:
            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    resp = await client.post(
                        f"http://{target_ip}:{target_port}/api/sync/pull-blob",
                        json={
                            "digest": blob.digest, "size": blob.size,
                            "source_ip": this_node_ip, "source_port": this_node_port,
                        },
                        timeout=None,
                    )
                    ok = resp.status_code == 200 and resp.json().get("success")
                    if on_progress:
                        await on_progress(SyncProgress(
                            model_id=model_id, blob_digest=blob.digest,
                            blob_index=index, total_blobs=total,
                            bytes_written=blob.size if ok else 0, blob_total_bytes=blob.size,
                            status="complete" if ok else "error",
                        ))
                    return ok
                except Exception as e:
                    logger.error("Failed to trigger blob transfer for %s: %s", blob.digest, e)
                    return False

    results = await asyncio.gather(*[
        transfer_one(b, i + 1) for i, b in enumerate(to_transfer)
    ])

    all_ok = all(results)
    if all_ok and manifest:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"http://{target_ip}:{target_port}/api/sync/write-manifest",
                json={"model_id": model_id, "manifest": manifest},
            )
    return all_ok
