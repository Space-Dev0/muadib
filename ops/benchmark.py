"""CPU renderer-to-localhost TCP benchmark. No GPU, LAN or trainer claims."""
import argparse
import json
import platform
import socket
import threading
import time
from pathlib import Path
from ops.metrics import Metrics
from ops.scene_stream import produce, consume


def run(count: int, batch_size: int, seed: int) -> dict:
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0)); listener.listen(1)
    port = listener.getsockname()[1]; errors = []
    def sender():
        try:
            conn, _ = listener.accept()
            with conn: produce(conn, count=count, batch_size=batch_size, seed=seed)
        except Exception as e: errors.append(str(e))
        finally: listener.close()
    thread = threading.Thread(target=sender); thread.start()
    m = Metrics(); start = time.monotonic()
    with socket.create_connection(('127.0.0.1', port), timeout=30) as conn:
        it = iter(consume(conn))
        while True:
            wait = time.monotonic()
            try: images, _, size = next(it)
            except StopIteration: break
            m.receive_decode_seconds += time.monotonic() - wait
            m.scenes += len(images); m.payload_bytes += size
    m.elapsed_seconds = time.monotonic() - start
    thread.join(30)
    if thread.is_alive() or errors or m.scenes != count: raise RuntimeError(f'benchmark failure: {errors}')
    return {'batch_size': batch_size, **m.result()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--seed', type=int, default=90000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1: parser.error('count must be positive')
    result = {'protocol': 'TCP, localhost, producer and consumer threads, PNG + JSON',
              'platform': platform.platform(), 'python': platform.python_version(),
              'count_per_batch': args.count, 'seed': args.seed,
              'measurements': [run(args.count, batch, args.seed) for batch in (2, 8, 16)],
              'limitations': 'CPU rendering and PNG encode/decode included; throughput is application payload, not link capacity; no GPU idle, LAN, trainer or training measurement.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
