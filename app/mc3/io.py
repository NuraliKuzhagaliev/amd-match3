"""Small framed protocol; no HTTP port, pickle, or model reload per query."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import time

MAX_MESSAGE = 16 * 1024 * 1024


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_exact(sock: socket.socket, size: int) -> bytes:
    parts = bytearray()
    while len(parts) < size:
        data = sock.recv(size - len(parts))
        if not data:
            raise ConnectionError("Worker closed an incomplete message")
        parts.extend(data)
    return bytes(parts)


def receive(sock: socket.socket) -> dict:
    size = struct.unpack("!I", read_exact(sock, 4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid protocol message size")
    value = json.loads(read_exact(sock, size))
    if not isinstance(value, dict):
        raise ValueError("Protocol message must be an object")
    return value


def send(sock: socket.socket, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(data) > MAX_MESSAGE:
        raise ValueError("Protocol message too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


def request(address: str, payload: dict, timeout: float, wait_ready: bool = False) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Worker was not ready before the deadline")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(remaining)
        try:
            sock.connect(address)
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            if not wait_ready:
                raise
            time.sleep(min(0.2, remaining))
            continue
        try:
            send(sock, payload)
            result = receive(sock)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result
        finally:
            sock.close()
