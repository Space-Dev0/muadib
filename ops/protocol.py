"""Bounded length-prefixed messages for trusted experimental TCP peers."""
import json
import socket
import struct
import time

MAX_HEADER = 65536
MAX_PAYLOAD = 16 * 1024 * 1024


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('request deadline exceeded')
    return seconds


def recv_exact(sock: socket.socket, size: int, *, deadline: float | None = None) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if deadline is not None:
            sock.settimeout(remaining(deadline))
        data = sock.recv(min(65536, size - len(chunks)))
        if not data:
            raise EOFError('connection closed before frame completed')
        chunks.extend(data)
    return bytes(chunks)


def send_json(sock: socket.socket, value: dict) -> None:
    data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(data) > MAX_HEADER:
        raise ValueError('header too large')
    sock.sendall(struct.pack('!I', len(data)) + data)


def recv_json(sock: socket.socket, *, deadline: float | None = None) -> dict:
    size = struct.unpack('!I', recv_exact(sock, 4, deadline=deadline))[0]
    if size > MAX_HEADER:
        raise ValueError('header too large')
    value = json.loads(recv_exact(sock, size, deadline=deadline))
    if not isinstance(value, dict):
        raise ValueError('expected object')
    return value
