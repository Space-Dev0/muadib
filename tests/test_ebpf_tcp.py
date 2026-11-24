import subprocess
import unittest
from unittest.mock import patch

from ops.ebpf_tcp import collect_retransmissions


class RetransmissionCollectorTests(unittest.TestCase):
    def test_port_and_duration_are_bounded_integers(self):
        for kwargs in ({"port": 0}, {"port": 65536}, {"port": True},
                       {"port": 22, "seconds": 0}, {"port": 22, "seconds": 61}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collect_retransmissions(**kwargs)

    @patch("ops.ebpf_tcp.platform.system", return_value="Windows")
    def test_windows_reports_unavailable_without_spawning(self, _):
        with patch("ops.ebpf_tcp.subprocess.run") as run:
            result = collect_retransmissions(port=22)
        self.assertFalse(result["available"])
        self.assertIsNone(result["tcp_retransmit_events"])
        run.assert_not_called()

    @patch("ops.ebpf_tcp.platform.system", return_value="Linux")
    @patch("ops.ebpf_tcp.shutil.which", return_value=None)
    def test_missing_bpftrace_is_not_zero(self, *_):
        result = collect_retransmissions(port=22)
        self.assertFalse(result["available"])
        self.assertIsNone(result["tcp_retransmit_events"])

    @patch("ops.ebpf_tcp.platform.system", return_value="Linux")
    @patch("ops.ebpf_tcp.shutil.which", return_value="/usr/bin/bpftrace")
    @patch("ops.ebpf_tcp.subprocess.run")
    def test_successful_capture_and_empty_capture(self, run, *_):
        for suffix, expected in (("\n@retransmits: 12\n", 12), ("\n", 0)):
            run.return_value = subprocess.CompletedProcess([], 0,
                "MUADIB_TCP_BEGIN\nMUADIB_TCP_END" + suffix, "")
            result = collect_retransmissions(port=2229, seconds=3)
            self.assertTrue(result["available"])
            self.assertEqual(result["tcp_retransmit_events"], expected)
            self.assertIsNone(result["dropped_events"])
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ["/usr/bin/bpftrace", "-q", "-e"])
        self.assertIn("args->sport == 2229", argv[3])
        self.assertIn("interval:s:3", argv[3])
        self.assertNotIn("shell", run.call_args.kwargs)

    @patch("ops.ebpf_tcp.platform.system", return_value="Linux")
    @patch("ops.ebpf_tcp.shutil.which", return_value="/usr/bin/bpftrace")
    @patch("ops.ebpf_tcp.subprocess.run")
    def test_failures_and_incomplete_output_are_unavailable(self, run, *_):
        cases = [
            subprocess.CompletedProcess([], 1, "", "permission denied"),
            subprocess.CompletedProcess([], 0, "MUADIB_TCP_BEGIN\n", ""),
            subprocess.CompletedProcess([], 0, "MUADIB_TCP_BEGIN\nMUADIB_TCP_END\n", "warning"),
            subprocess.CompletedProcess([], 0, "MUADIB_TCP_BEGIN\nMUADIB_TCP_END\n@retransmits: 1\n@retransmits: 2\n", ""),
        ]
        for completed in cases:
            run.return_value = completed
            result = collect_retransmissions(port=22)
            self.assertFalse(result["available"])
            self.assertIsNone(result["tcp_retransmit_events"])
            self.assertTrue(result["error"])
        run.side_effect = subprocess.TimeoutExpired("bpftrace", 20)
        result = collect_retransmissions(port=22)
        self.assertFalse(result["available"])
        self.assertIsNone(result["tcp_retransmit_events"])
        self.assertIsNotNone(result["collector_elapsed_seconds"])


if __name__ == "__main__":
    unittest.main()
