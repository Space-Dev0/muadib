"""Request-driven conditional renderer transport, not a trainer integration.

Trusted peers only: no authentication or encryption. Each request contains
rule names and per-scene RNG seeds chosen by the future trainer; the producer
must use the same ordered rule definitions and source versions. A checksum
identifies exact canonical definitions; it does not authenticate either peer.
"""
from dataclasses import asdict
import hashlib
import io
import json
import socket
import sys
import time
from pathlib import Path
import PIL
from exposure import render as render_module, rules as rules_module, measure as measure_module
from PIL import Image
from exposure.measure import load_rules
from exposure.render import RenderedScene, render_conditioned
from exposure.rules import SceneObject, satisfies
from ops.protocol import MAX_PAYLOAD, recv_exact, recv_json, send_json

MAX_BATCH = 128


def definitions_hash(definitions: dict) -> str:
    # Insertion order matters to the pilot's token indices and RNG choices.
    return hashlib.sha256(json.dumps(definitions, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def renderer_runtime() -> dict:
    """Fingerprint the remote scene distribution, drawing code, and RNG runtime."""
    paths = {'render': Path(render_module.__file__), 'rules': Path(rules_module.__file__),
             'measure': Path(measure_module.__file__)}
    return {'python': list(sys.version_info[:3]), 'pillow': PIL.__version__,
            'code_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in paths.items()}}


def serve_requests(sock: socket.socket, definitions: dict) -> None:
    rules = load_rules(definitions)
    fingerprint = definitions_hash(definitions)
    while True:
        request = recv_json(sock)
        if request.get('kind') == 'close': return
        jobs, max_attempts = request.get('jobs'), request.get('max_attempts')
        if (request.get('kind') != 'render' or request.get('rules_sha256') != fingerprint
                or not isinstance(jobs, list) or not 1 <= len(jobs) <= MAX_BATCH
                or type(max_attempts) is not int or not 1 <= max_attempts <= 1_000_000
                or any(not isinstance(job, dict) or set(job) != {'rule', 'seed'}
                       or job['rule'] not in rules or type(job['seed']) is not int
                       or not 0 <= job['seed'] < 2**63 for job in jobs)):
            raise ValueError('invalid render request or incompatible rule definitions')
        records, images = [], []
        for job in jobs:
            scene, attempts = render_conditioned(job['seed'], rules[job['rule']], max_attempts)
            data = io.BytesIO(); scene.image.save(data, format='PNG'); payload = data.getvalue()
            images.append(payload)
            records.append({**job, 'attempts': attempts, 'objects': [asdict(o) for o in scene.objects],
                            'png_bytes': len(payload), 'png_sha256': hashlib.sha256(payload).hexdigest()})
        if sum(map(len, images)) > MAX_PAYLOAD: raise ValueError('response too large')
        send_json(sock, {'kind': 'rendered', 'rules_sha256': fingerprint, 'runtime': renderer_runtime(), 'records': records})
        for image in images: sock.sendall(image)


class ConditionalRendererClient:
    """Trainer-facing adapter; call render_batch with explicitly chosen jobs."""
    def __init__(self, sock: socket.socket, definitions: dict):
        self.sock = sock
        self.rules = load_rules(definitions)
        self.fingerprint = definitions_hash(definitions)
        self.renderer_runtime = renderer_runtime()
        self.render_seconds = 0.0
        self.png_bytes = 0

    def render_batch(self, jobs: list[dict], max_attempts: int) -> list[tuple[RenderedScene, int]]:
        started = time.perf_counter()
        send_json(self.sock, {'kind': 'render', 'jobs': jobs, 'max_attempts': max_attempts,
                              'rules_sha256': self.fingerprint})
        result = recv_json(self.sock); records = result.get('records')
        if (result.get('kind') != 'rendered' or result.get('rules_sha256') != self.fingerprint
                or result.get('runtime') != self.renderer_runtime
                or not isinstance(records, list) or len(records) != len(jobs)):
            raise ValueError('invalid render response')
        output = []; total = 0
        for job, record in zip(jobs, records, strict=True):
            if (not isinstance(record, dict) or record.get('rule') != job['rule']
                    or record.get('seed') != job['seed'] or type(record.get('attempts')) is not int
                    or not 1 <= record['attempts'] <= max_attempts
                    or type(record.get('png_bytes')) is not int or not 1 <= record['png_bytes'] <= MAX_PAYLOAD):
                raise ValueError('invalid rendered record')
            total += record['png_bytes']
            if total > MAX_PAYLOAD: raise ValueError('response too large')
            raw = recv_exact(self.sock, record['png_bytes'])
            if hashlib.sha256(raw).hexdigest() != record.get('png_sha256'):
                raise ValueError('corrupt PNG payload')
            im = Image.open(io.BytesIO(raw))
            if im.size != (64,64) or im.format != 'PNG': raise ValueError('invalid image')
            im.load()
            try:
                objects = tuple(SceneObject(**item) for item in record['objects'])
                if not satisfies(objects, self.rules[job['rule']]): raise ValueError('condition mismatch')
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('invalid scene annotations') from exc
            output.append((RenderedScene(im.convert('RGB'), objects), record['attempts']))
        self.render_seconds += time.perf_counter() - started
        self.png_bytes += total
        return output

    def close(self):
        send_json(self.sock, {'kind': 'close'})
