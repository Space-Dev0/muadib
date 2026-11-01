"""Report local GPU and optional PyTorch CUDA availability without training.

This standard-library-only command is intended to run on the target machine;
it does not install packages or assume that an NVIDIA driver implies PyTorch
CUDA support.
"""

import csv
import importlib
import importlib.util
import io
import json
import platform
import subprocess
import sys


def nvidia_info() -> dict:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except FileNotFoundError:
        return {"status": "not_found", "devices": []}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "devices": [], "error": str(exc)}
    if result.returncode != 0:
        return {"status": "error", "devices": [], "error": result.stderr.strip() or
                f"nvidia-smi exited with code {result.returncode}"}
    try:
        rows = list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
        if not rows or any(len(row) != 3 for row in rows):
            raise ValueError("Expected three fields for each GPU")
        devices = []
        for name, memory, driver in rows:
            memory_mib = int(memory.strip())
            if not name.strip() or not driver.strip() or memory_mib <= 0:
                raise ValueError("Invalid GPU fields")
            devices.append({"name": name.strip(), "memory_mib": memory_mib, "driver_version": driver.strip()})
    except (ValueError, csv.Error) as exc:
        return {"status": "error", "devices": [], "error": f"Invalid nvidia-smi output: {exc}"}
    return {"status": "ok", "devices": devices}


def pytorch_info() -> dict:
    try:
        if importlib.util.find_spec("torch") is None:
            return {"status": "not_installed"}
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        return {"status": "ok", "version": str(torch.__version__),
                "compiled_cuda_version": torch.version.cuda, "cuda_available": available,
                "cuda_device_count": int(torch.cuda.device_count()) if available else 0}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def report() -> dict:
    return {"schema_version": 1,
            "system": {"os": platform.system(), "os_release": platform.release(),
                       "architecture": platform.machine(), "python_version": platform.python_version()},
            "nvidia": nvidia_info(), "pytorch": pytorch_info(),
            "note": "Read-only preflight; no training, memory-fit estimate, or Stage 0 validation."}


def main() -> None:
    try:
        print(json.dumps(report(), indent=2, allow_nan=False))
    except BrokenPipeError:
        sys.exit(1)


if __name__ == "__main__":
    main()
