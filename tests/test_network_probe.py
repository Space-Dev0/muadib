"""Exercise real local TCP echo and loss accounting without a network daemon."""
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ops.network_probe import EchoServer, _percentile, main, probe


class NetworkProbeTests(unittest.TestCase):
    def test_loopback_echo_and_latency_summary(self):
        with EchoServer(0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = probe(server.server_address[1], count=3, interval=0,
                               timeout=1, payload_size=32)
            finally:
                server.shutdown()
                thread.join(timeout=2)
        summary = result["summary"]
        self.assertEqual((summary["attempts"], summary["successes"], summary["failed_attempts"]), (3, 3, 0))
        self.assertEqual(summary["rtt_variation_count"], 2)
        self.assertGreaterEqual(summary["echo_rtt_median_ms"], 0)
        self.assertGreater(summary["echoed_payload_goodput_mbps"], 0)
        self.assertEqual([a["sequence"] for a in result["attempts"]], [0, 1, 2])
        self.assertEqual(result["configuration"]["payload_pattern"], "fresh_random_bytes_per_attempt")
        self.assertIsNone(result["configuration"]["server_socket_timeout_seconds"])
        self.assertTrue(any("nearest-rank" in note for note in result["notes"]))

    def test_unavailable_endpoint_is_retained_as_failure(self):
        with socket.socket() as temporary:
            temporary.bind(("127.0.0.1", 0))
            port = temporary.getsockname()[1]
        result = probe(port, count=2, interval=0, timeout=0.2)
        self.assertEqual(result["summary"]["successes"], 0)
        self.assertEqual(result["summary"]["failure_fraction"], 1)
        self.assertIsNone(result["summary"]["echo_rtt_p95_ms"])
        self.assertEqual(result["summary"]["rtt_variation_count"], 0)
        self.assertEqual(len(result["attempts"]), 2)

    def test_deadline_does_not_restart_after_each_chunk(self):
        class Clock:
            value = 0.0
            def monotonic(self):
                self.value += 0.03
                return self.value
        class SlowSocket:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def settimeout(self, timeout): pass
            def sendall(self, value): pass
            def recv(self, count): return b'x'
        clock = Clock()
        with patch('ops.network_probe.time.monotonic', side_effect=clock.monotonic), \
             patch('ops.network_probe.socket.create_connection', return_value=SlowSocket()):
            result = probe(1234, count=1, interval=0, timeout=0.15, payload_size=32)
        self.assertEqual(result['attempts'][0]['error'], 'TimeoutError')
        self.assertEqual(result['summary']['failed_attempts'], 1)

    def test_zero_elapsed_goodput_is_unknown(self):
        with patch('ops.network_probe.time.monotonic', return_value=1.0), \
             patch('ops.network_probe.socket.create_connection', side_effect=ConnectionRefusedError()):
            result = probe(1234, count=1, interval=0)
        self.assertIsNone(result['summary']['echoed_payload_goodput_mbps'])
        self.assertEqual(result['summary']['failed_attempts'], 1)

    def test_slow_drip_obeys_total_attempt_deadline(self):
        try:
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
        except PermissionError as exc:
            self.skipTest(str(exc))
        def slow_peer():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.recv(4096)
                    for _ in range(20):
                        connection.sendall(b'x')
                        time.sleep(0.04)
            except OSError:
                pass
            finally:
                listener.close()
        thread = threading.Thread(target=slow_peer, daemon=True)
        thread.start()
        start = time.monotonic()
        result = probe(listener.getsockname()[1], count=1, interval=0, timeout=0.15, payload_size=32)
        elapsed = time.monotonic() - start
        thread.join(timeout=2)
        self.assertEqual(result['attempts'][0]['status'], 'error')
        self.assertEqual(result['attempts'][0]['error'], 'TimeoutError')
        self.assertLess(elapsed, 0.6)

    def test_parameter_validation_and_nearest_rank(self):
        for kwargs in ({"count": 0}, {"timeout": float("nan")}, {"payload_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                probe(1234, **kwargs)
        self.assertEqual(_percentile([3, 1, 2], 0.5), 2)
        self.assertEqual(_percentile([3, 1, 2], 0.95), 3)
        self.assertIsNone(_percentile([], 0.5))
        self.assertEqual(_percentile(list(range(1, 21)), 0.5), 10)

    def test_invalid_output_is_rejected_before_measuring(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'occupied.json'
            output.write_text('keep me')
            with patch('ops.network_probe.probe') as measure, self.assertRaises(SystemExit) as cm:
                main(['probe', '--port', '1234', '--output', str(output)])
            self.assertEqual(cm.exception.code, 1)
            measure.assert_not_called()
            self.assertEqual(output.read_text(), 'keep me')

            parent_file = Path(tmp) / 'not-a-dir'
            parent_file.write_text('untouched')
            with patch('ops.network_probe.probe') as measure, self.assertRaises(SystemExit):
                main(['probe', '--port', '1234', '--output', str(parent_file / 'out.json')])
            measure.assert_not_called()
            self.assertEqual(parent_file.read_text(), 'untouched')

    def test_failed_probe_removes_own_empty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new.json'
            with patch('ops.network_probe.probe', side_effect=ValueError('bad config')), self.assertRaises(SystemExit):
                main(['probe', '--port', '1234', '--output', str(output)])
            self.assertFalse(output.exists())

    def test_server_timeout_validation(self):
        for timeout in (0, -1, float('nan'), 3601):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                EchoServer(0, socket_timeout=timeout)


if __name__ == "__main__":
    unittest.main()
