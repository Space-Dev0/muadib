"""Unconditioned renderer -> TCP batches with JSON annotations and PNG bytes.

Experimental transport only. The pilot trainer is not connected to this stream.
Run only on a trusted network: frames have no authentication or encryption.
"""
import argparse
from dataclasses import asdict
import io
import json
import socket
import struct
import time
from pathlib import Path
from PIL import Image
from exposure.render import render_scene
from ops.metrics import Metrics, append_jsonl
from ops.protocol import MAX_PAYLOAD, recv_exact, recv_json, send_json


def produce(sock: socket.socket, *, count: int, batch_size: int, seed: int = 0) -> None:
    if count < 0 or not 1 <= batch_size <= 1024:
        raise ValueError('invalid count or batch size')
    for start in range(0, count, batch_size):
        images, annotations = [], []
        for i in range(start, min(start + batch_size, count)):
            result = render_scene(seed + i)
            buf = io.BytesIO()
            result.image.save(buf, format='PNG')
            images.append(buf.getvalue())
            annotations.append({'seed': seed + i, 'objects': [asdict(obj) for obj in result.objects]})
        sizes = [len(x) for x in images]
        send_json(sock, {'kind': 'batch', 'sizes': sizes, 'annotations': annotations})
        sock.sendall(b''.join(images))
    send_json(sock, {'kind': 'end'})


def consume(sock: socket.socket):
    """Yield (list of RGB PIL images, list of annotations, image payload bytes)."""
    while True:
        msg = recv_json(sock)
        if msg.get('kind') == 'end':
            return
        sizes, annotations = msg.get('sizes'), msg.get('annotations')
        if (msg.get('kind') != 'batch' or not isinstance(sizes, list) or not sizes
                or not isinstance(annotations, list) or len(sizes) != len(annotations)
                or any(type(s) is not int or s <= 0 for s in sizes) or sum(sizes) > MAX_PAYLOAD):
            raise ValueError('invalid scene frame')
        images = []
        for size in sizes:
            image = Image.open(io.BytesIO(recv_exact(sock, size)))
            if image.size != (64, 64) or image.format != 'PNG':
                raise ValueError('unexpected image format or size')
            image.load()
            images.append(image.convert('RGB'))
        yield images, annotations, sum(sizes)


def serve(host: str, port: int, count: int, batch_size: int, seed: int) -> None:
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port)); server.listen(1)
        conn, _ = server.accept()
        with conn:
            conn.settimeout(30)
            produce(conn, count=count, batch_size=batch_size, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['serve', 'receive'])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=53121)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--metrics', type=Path)
    args = parser.parse_args()
    if args.mode == 'serve':
        serve(args.host, args.port, args.count, args.batch_size, args.seed)
    else:
        metrics = Metrics(); start = time.monotonic()
        with socket.create_connection((args.host, args.port), timeout=30) as conn:
            iterator = iter(consume(conn))
            while True:
                wait = time.monotonic()
                try:
                    images, _, size = next(iterator)
                except StopIteration:
                    break
                metrics.receive_decode_seconds += time.monotonic() - wait
                metrics.scenes += len(images); metrics.payload_bytes += size
        metrics.elapsed_seconds = time.monotonic() - start
        result = metrics.result(); result.update({'host': args.host, 'batch_size': args.batch_size,
                                                   'note': 'CPU-rendered PNG payload; no GPU idle measurement'})
        print(json.dumps(result, indent=2))
        if args.metrics: append_jsonl(args.metrics, result)


if __name__ == '__main__': main()
