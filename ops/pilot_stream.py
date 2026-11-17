"""Opt-in pilot trainer adapter over a verified conditional scene connection.

This wrapper leaves exposure.pilot_train unchanged. It temporarily replaces its
renderer in this process, therefore concurrent pilot calls are forbidden. A
version-4 wrapper checkpoint cannot be resumed by the normal v3 trainer.
Uses a trusted or separately authenticated connection, not a public endpoint.
"""
import hashlib
from pathlib import Path
import threading
from ops.conditional_stream import ConditionalRendererClient, renderer_runtime
from ops import conditional_stream, protocol

_LOCK = threading.Lock()


def _fingerprint() -> dict:
    paths = {'adapter': Path(__file__), 'conditional_stream': Path(conditional_stream.__file__),
             'protocol': Path(protocol.__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def train_via_stream(definitions: dict, weights: object | None, sock, *, resume=None, **kwargs):
    """Run pilot updates using explicitly seeded remote scenes.

    Existing v3 checkpoints are deliberately rejected. Resume only a v4 stream
    checkpoint through this wrapper and the same verified renderer/transport.
    """
    import exposure.pilot_train as pilot
    from exposure.measure import load_rules
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('concurrent streamed pilot training is unsupported')
    original = pilot.render_conditioned
    try:
        rules = load_rules(definitions)
        # Equivalent duplicate rules may share a renderer name; counts remain
        # associated with the pilot's original selected condition index.
        rule_by_predicates = {rule: name for name, rule in rules.items()}
        client = ConditionalRendererClient(sock, definitions)
        provenance = {'schema_version': 1, 'rules_sha256': client.fingerprint,
                      'renderer_runtime': renderer_runtime(), 'transport_sha256': _fingerprint()}
        local_resume = None
        if resume is not None:
            if not isinstance(resume, dict) or resume.get('schema_version') != 4 or resume.get('stream') != provenance:
                raise ValueError('stream checkpoint provenance does not match current code and renderer')
            local_resume = dict(resume)
            local_resume['schema_version'] = 3
            local_resume.pop('stream', None)
        def render_remote(seed, predicates, max_attempts=10000):
            try: name = rule_by_predicates[predicates]
            except KeyError as exc: raise ValueError('unknown remote condition') from exc
            return client.render_batch([{'rule': name, 'seed': seed}], max_attempts)[0]
        pilot.render_conditioned = render_remote
        checkpoint, report = pilot.train(definitions, weights, resume=local_resume, **kwargs)
        checkpoint['schema_version'] = 4
        checkpoint['stream'] = provenance
        report['scene_transport'] = 'conditional_tcp_experimental'
        report['current_segment_remote_render_seconds'] = client.render_seconds
        report['current_segment_png_payload_bytes'] = client.png_bytes
        report['note'] += ' Experimental external renderer; source/runtime verified; unauthenticated transport.'
        return checkpoint, report
    finally:
        pilot.render_conditioned = original
        _LOCK.release()
