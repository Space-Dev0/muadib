import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from ops.checkpoint_sync import receive_one, send_file, sha256
from ops.scene_stream import consume, produce


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            a, b = socket.socketpair()
            try:
                a.sendall(b"x")
                b.recv(1)
            finally:
                a.close(); b.close()
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
        except PermissionError as exc:
            raise unittest.SkipTest(f'local sockets unavailable: {exc}') from exc

    def test_scene_batches_roundtrip(self):
        a, b = socket.socketpair()
        errors = []
        def worker():
            try:
                with a: produce(a, count=9, batch_size=4, seed=31)
            except Exception as exc: errors.append(exc)
        thread = threading.Thread(target=worker); thread.start()
        with b: batches = list(consume(b))
        thread.join(10)
        self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
        self.assertEqual([len(x[0]) for x in batches], [4, 4, 1])
        self.assertEqual([a['seed'] for _, ann, _ in batches for a in ann], list(range(31, 40)))
        self.assertTrue(all(im.size == (64, 64) for ims, _, _ in batches for im in ims))

    def test_conditional_render_matches_local_seeded_renderer(self):
        from ops.conditional_stream import ConditionalRendererClient, serve_requests
        from exposure.render import render_conditioned
        from exposure.measure import load_rules
        definitions = {'red': [{'type': 'attribute', 'variable': 'x', 'field': 'color', 'value': 'red'}],
                       'circle': [{'type': 'attribute', 'variable': 'x', 'field': 'shape', 'value': 'circle'}]}
        jobs = [{'rule': 'red', 'seed': 123}, {'rule': 'circle', 'seed': 456}]
        a, b = socket.socketpair(); errors = []
        def worker():
            try:
                with a: serve_requests(a, definitions)
            except Exception as exc: errors.append(exc)
        thread = threading.Thread(target=worker); thread.start()
        with b:
            client = ConditionalRendererClient(b, definitions)
            results = client.render_batch(jobs, 10000); client.close()
        thread.join(10)
        self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
        rules = load_rules(definitions)
        for job, (scene, attempts) in zip(jobs, results, strict=True):
            expected, expected_attempts = render_conditioned(job['seed'], rules[job['rule']], 10000)
            self.assertEqual(attempts, expected_attempts)
            self.assertEqual(scene.objects, expected.objects)
            self.assertEqual(scene.image.tobytes(), expected.image.tobytes())

    def test_transfer_resumes_and_publishes_identical_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root / 'src'; dest = root / 'dest'
            src.write_bytes(os.urandom(300_000))
            listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen()
            port = listener.getsockname()[1]; outcomes = []; errors = []
            def worker():
                try:
                    for _ in range(2):
                        conn, _ = listener.accept()
                        with conn: outcomes.append(receive_one(conn, dest))
                except Exception as exc: errors.append(exc)
                finally: listener.close()
            thread = threading.Thread(target=worker); thread.start()
            result = send_file('127.0.0.1', port, src, simulate_drop_after=50_000)
            thread.join(10)
            self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
            self.assertEqual(outcomes, [False, True]); self.assertEqual(dest.read_bytes(), src.read_bytes())
            self.assertEqual(result['sha256'], sha256(dest))
            self.assertEqual(result['resume_offsets'], [0, 50_000])

    def test_corrupted_compressed_payload_is_not_published(self):
        from ops.protocol import send_json, recv_json
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / 'checkpoint'
            a, b = socket.socketpair(); errors = []
            def receiver():
                try:
                    with b: receive_one(b, destination)
                except Exception as exc: errors.append(exc)
            thread = threading.Thread(target=receiver); thread.start()
            with a:
                send_json(a, {'version': 1, 'sha256': '0'*64, 'bytes': 4,
                              'compressed_sha256': '1'*64, 'compressed_bytes': 4})
                self.assertEqual(recv_json(a)['offset'], 0)
                a.sendall(b'oops')
            thread.join(10)
            self.assertFalse(thread.is_alive()); self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ValueError)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_existing_destination_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / 'checkpoint'; destination.write_bytes(b'original')
            a, b = socket.socketpair()
            from ops.protocol import send_json
            with a, b:
                send_json(a, {'version': 1, 'sha256': '0'*64, 'bytes': 4,
                              'compressed_sha256': '1'*64, 'compressed_bytes': 4})
                with self.assertRaises(FileExistsError): receive_one(b, destination)
            self.assertEqual(destination.read_bytes(), b'original')


if __name__ == '__main__': unittest.main()
