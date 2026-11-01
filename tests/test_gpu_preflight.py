import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from exposure.gpu_preflight import nvidia_info, pytorch_info


class GpuPreflightTests(unittest.TestCase):
    @patch("exposure.gpu_preflight.subprocess.run")
    def test_nvidia_query_parses_multiple_devices(self, run):
        run.return_value = subprocess.CompletedProcess([], 0,
            'NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144, 581.83\n"Other, GPU", 8192, 581.83\n', '')
        result = nvidia_info()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["devices"][0]["memory_mib"], 6144)
        self.assertEqual(result["devices"][1]["name"], "Other, GPU")

    @patch("exposure.gpu_preflight.subprocess.run")
    def test_missing_failed_and_invalid_nvidia_command(self, run):
        run.side_effect = FileNotFoundError()
        self.assertEqual(nvidia_info()["status"], "not_found")
        run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 15)
        self.assertEqual(nvidia_info()["status"], "error")
        run.side_effect = None
        run.return_value = subprocess.CompletedProcess([], 1, '', 'driver unavailable')
        self.assertEqual(nvidia_info()["error"], 'driver unavailable')
        run.return_value = subprocess.CompletedProcess([], 0, 'invalid', '')
        self.assertEqual(nvidia_info()["status"], "error")

    @patch("exposure.gpu_preflight.importlib.util.find_spec", return_value=None)
    def test_torch_is_optional(self, _find):
        self.assertEqual(pytorch_info(), {"status": "not_installed"})

    @patch("exposure.gpu_preflight.importlib.import_module")
    @patch("exposure.gpu_preflight.importlib.util.find_spec", return_value=True)
    def test_torch_cuda_status_and_import_errors(self, _find, load):
        torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None),
                                cuda=SimpleNamespace(is_available=lambda: False))
        load.return_value = torch
        self.assertEqual(pytorch_info()["cuda_available"], False)
        self.assertEqual(pytorch_info()["cuda_device_count"], 0)
        torch.version.cuda = "test-cuda"
        torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        self.assertEqual(pytorch_info()["cuda_device_count"], 1)
        load.side_effect = OSError("bad installation")
        self.assertEqual(pytorch_info()["status"], "error")

    def test_cli_emits_json_without_training_dependencies(self):
        result = subprocess.run([sys.executable, "-m", "exposure.gpu_preflight"],
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertIn(report["nvidia"]["status"], {"ok", "error", "not_found"})
        self.assertIn(report["pytorch"]["status"], {"ok", "error", "not_installed"})


if __name__ == "__main__":
    unittest.main()
