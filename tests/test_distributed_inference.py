import io
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
import json
import struct

from PIL import Image
from ops.distributed_inference import digest, handle, run, select_checkpoint_format


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, "PNG")
    return buffer.getvalue()


class InferenceTests(unittest.TestCase):
    def test_loopback_failover_and_checksum(self):
        try:
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
        except PermissionError as exc:
            self.skipTest(str(exc))
        listener.listen()
        good = listener.getsockname()
        checkpoint_hash = digest(b"checkpoint")

        def worker():
            try:
                connection, _ = listener.accept()
                with connection:
                    handle(connection, checkpoint_hash, lambda color, seed: png())
            finally:
                listener.close()

        thread = threading.Thread(target=worker)
        thread.start()
        job = {"checkpoint_sha256": checkpoint_hash, "color": "red", "seed": 7}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            # Port zero cannot have a listener, forcing the first request to fail.
            records = run([job], [("127.0.0.1", 0), good], output, timeout=5)
            self.assertEqual(records[0]["worker_index"], 1)
            self.assertEqual(records[0]["failed_attempts"], 1)
            self.assertEqual(records[0]["sha256"], digest((output / records[0]["file"]).read_bytes()))
            self.assertTrue((output / "manifest.json").is_file())
        thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_hash_mismatch_and_duplicate_rejection(self):
        job = {"checkpoint_sha256": digest(b"different"), "color": "red", "seed": 1}
        encoded = json.dumps(job).encode()
        class FakeSocket:
            def __init__(self):
                self.input = io.BytesIO(struct.pack('!I', len(encoded)) + encoded)
                self.output = bytearray()
            def recv(self, size): return self.input.read(size)
            def sendall(self, data): self.output.extend(data)
        sock = FakeSocket()
        handle(sock, digest(b"expected"), lambda *_: png())
        self.assertEqual(json.loads(sock.output[4:])["status"], "error")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run([job, job], [("127.0.0.1", 1)], Path(directory) / "out")

    def test_failover_without_network(self):
        job = {"checkpoint_sha256": digest(b"checkpoint"), "color": "red", "seed": 2}
        def fake_request(endpoint, _job, _timeout):
            if endpoint[1] == 1:
                raise OSError("disconnected")
            return png(), 0.1, {"device": "fake"}
        with tempfile.TemporaryDirectory() as directory, patch(
                "ops.distributed_inference.request", side_effect=fake_request):
            output = Path(directory) / "out"
            records = run([job], [("127.0.0.1", 1), ("127.0.0.1", 2)], output)
            self.assertEqual(records[0]["failed_attempts"], 1)
            self.assertEqual(records[0]["sha256"], digest((output / records[0]["file"]).read_bytes()))
            self.assertEqual(records[0]["runtime"], {"device": "fake"})

    def test_resume_partial_and_reject_changed_or_tampered_work(self):
        jobs = [{"checkpoint_sha256": digest(b"checkpoint"), "color": "red", "seed": seed}
                for seed in (1, 2)]
        def fail_second(endpoint, job, timeout):
            if job["seed"] == 2:
                raise EOFError("dropped")
            return png(), 0.1, {"device": "fake"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with patch("ops.distributed_inference.request", side_effect=fail_second):
                with self.assertRaises(RuntimeError):
                    run(jobs, [("127.0.0.1", 1)], output)
            self.assertTrue((output / "red-1" / "image.png").exists())
            self.assertFalse((output / "manifest.json").exists())
            with patch("ops.distributed_inference.request", return_value=(png(), 0.2, {})) as mock:
                records = run(jobs, [("127.0.0.1", 1)], output, resume=True)
                self.assertEqual(mock.call_count, 1)
                self.assertEqual(mock.call_args.args[1], jobs[1])
            self.assertEqual(len(records), 2)
            with self.assertRaises(ValueError):
                run(jobs[:1], [("127.0.0.1", 1)], output, resume=True)
            (output / "red-1" / "image.png").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                run(jobs, [("127.0.0.1", 1)], output, resume=True)

    def test_existing_lock_is_never_removed_by_second_coordinator(self):
        job = {"checkpoint_sha256": digest(b"checkpoint"), "color": "red", "seed": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            lock = output / ".coordinator.lock"
            lock.write_text("other process")
            with self.assertRaises(RuntimeError):
                run([job], [("127.0.0.1", 1)], output, resume=True)
            self.assertEqual(lock.read_text(), "other process")

    def test_checkpoint_format_selection(self):
        from types import SimpleNamespace
        probe = SimpleNamespace(load_checkpoint=object(), sample=object())
        warmstart = SimpleNamespace(load_warmstart_checkpoint=object(), sample=object())
        self.assertEqual(select_checkpoint_format("probe", probe, warmstart),
                         (probe.load_checkpoint, probe.sample))
        self.assertEqual(select_checkpoint_format("warmstart", probe, warmstart),
                         (warmstart.load_warmstart_checkpoint, warmstart.sample))
        with self.assertRaises(ValueError):
            select_checkpoint_format("unknown", probe, warmstart)


if __name__ == "__main__":
    unittest.main()
