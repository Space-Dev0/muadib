"""Bounded length-prefixed messages for trusted experimental TCP peers."""
import json
import socket
import struct

MAX_HEADER = 65536
MAX_PAYLOAD = 16 * 1024 * 1024


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
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


def recv_json(sock: socket.socket) -> dict:
    size = struct.unpack('!I', recv_exact(sock, 4))[0]
    if size > MAX_HEADER:
        raise ValueError('header too large')
    value = json.loads(recv_exact(sock, size))
    if not isinstance(value, dict):
        raise ValueError('expected object')
    return value
