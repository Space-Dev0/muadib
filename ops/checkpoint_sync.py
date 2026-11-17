"""Resumable compressed checkpoint transfer for trusted TCP peers.

No transport authentication/encryption. Use only an authenticated tunnel or
isolated trusted network. SHA-256 detects corruption, not a malicious peer.
Receiver never overwrites an existing destination. Partial files are retained.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import time
from ops.protocol import recv_json, send_json

CHUNK = 65536
MAX_FILE = 8 * 1024**3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''): h.update(block)
    return h.hexdigest()


def receive_one(conn: socket.socket, destination: Path) -> bool:
    """Receive one connection. Return True only after a verified atomic publish."""
    header = recv_json(conn)
    compressed_hash, raw_hash = header.get('compressed_sha256'), header.get('sha256')
    total, raw_size = header.get('compressed_bytes'), header.get('bytes')
    if (header.get('version') != 1 or not all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h)
                                              for h in (compressed_hash, raw_hash))
            or type(total) is not int or not 0 < total <= MAX_FILE
            or type(raw_size) is not int or not 0 <= raw_size <= MAX_FILE):
        raise ValueError('invalid transfer metadata')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == raw_size and sha256(destination) == raw_hash:
            send_json(conn, {'done': True}); return True
        raise FileExistsError('destination exists and differs; refusing overwrite')
    part = destination.with_name(destination.name + '.' + compressed_hash + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > total: raise ValueError('partial file exceeds expected length')
    send_json(conn, {'offset': offset})
    with part.open('ab') as output:
        while offset < total:
            block = conn.recv(min(CHUNK, total - offset))
            if not block:
                return False
            output.write(block); offset += len(block)
        output.flush(); os.fsync(output.fileno())
    if sha256(part) != compressed_hash:
        part.unlink()
        raise ValueError('compressed transfer failed SHA-256 validation')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.',
                                         suffix='.unpacked', delete=False) as output:
            tmp = Path(output.name); h = hashlib.sha256(); decoded = 0
            with gzip.open(part, 'rb') as compressed:
                for block in iter(lambda: compressed.read(CHUNK), b''):
                    decoded += len(block)
                    if decoded > raw_size: raise ValueError('decompressed file exceeds declared size')
                    h.update(block); output.write(block)
            output.flush(); os.fsync(output.fileno())
        if decoded != raw_size or h.hexdigest() != raw_hash:
            raise ValueError('decompressed checkpoint failed SHA-256 validation')
        os.link(tmp, destination)  # exclusive publication: cannot overwrite an existing file
        part.unlink()
    finally:
        if tmp is not None: tmp.unlink(missing_ok=True)
    send_json(conn, {'done': True})
    return True


def send_file(host: str, port: int, source: Path, *, retries: int = 5,
              simulate_drop_after: int | None = None) -> dict:
    """Compress once, retry connections and resume from receiver's verified offset."""
    if retries < 1: raise ValueError('retries must be positive')
    size = source.stat().st_size
    if size > MAX_FILE: raise ValueError('checkpoint too large')
    raw_hash = sha256(source)
    with tempfile.TemporaryDirectory(prefix='muadib-sync-') as directory:
        packed = Path(directory) / 'payload.gz'
        with source.open('rb') as inp, packed.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as out:
                for chunk in iter(lambda: inp.read(CHUNK), b''): out.write(chunk)
        # Detect a changing checkpoint rather than transmitting an inconsistent snapshot.
        if source.stat().st_size != size or sha256(source) != raw_hash:
            raise ValueError('source changed during compression')
        packed_size = packed.stat().st_size
        header = {'version': 1, 'sha256': raw_hash, 'bytes': size,
                  'compressed_sha256': sha256(packed), 'compressed_bytes': packed_size}
        dropped = False; offsets = []
        for attempt in range(retries):
            try:
                with socket.create_connection((host, port), timeout=30) as conn:
                    send_json(conn, header); reply = recv_json(conn)
                    if reply.get('done') is True:
                        return {'sha256': raw_hash, 'bytes': size, 'compressed_bytes': packed_size,
                                'resume_offsets': offsets}
                    offset = reply.get('offset')
                    if type(offset) is not int or not 0 <= offset <= packed_size:
                        raise ValueError('receiver supplied invalid offset')
                    offsets.append(offset)
                    with packed.open('rb') as f:
                        f.seek(offset); sent = 0
                        while offset < packed_size:
                            limit = min(CHUNK, packed_size - offset)
                            if simulate_drop_after is not None and not dropped:
                                limit = min(limit, max(1, simulate_drop_after - sent))
                            block = f.read(limit)
                            conn.sendall(block); offset += len(block); sent += len(block)
                            if simulate_drop_after is not None and not dropped and sent >= simulate_drop_after:
                                dropped = True; break
                    if dropped and simulate_drop_after is not None:
                        simulate_drop_after = None
                        continue
                    if recv_json(conn).get('done') is not True:
                        raise ValueError('receiver did not confirm verified publication')
                    return {'sha256': raw_hash, 'bytes': size, 'compressed_bytes': packed_size,
                            'resume_offsets': offsets}
            except (OSError, EOFError):
                if attempt == retries - 1: raise
                time.sleep(.2)
    raise RuntimeError('transfer retries exhausted')


def serve(host: str, port: int, destination: Path) -> None:
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port)); listener.listen(4)
        while True:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(30)
                try:
                    if receive_one(conn, destination): return
                except (ValueError, EOFError, OSError) as exc:
                    print(f'Transfer rejected or interrupted: {exc}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['serve', 'send'])
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=53122)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--simulate-drop-after', type=int)
    args = parser.parse_args()
    if args.mode == 'serve': serve(args.host, args.port, args.file)
    else: print(json.dumps(send_file(args.host, args.port, args.file,
                                    simulate_drop_after=args.simulate_drop_after), indent=2))


if __name__ == '__main__': main()
