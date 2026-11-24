"""Shard fixed-seed MOSAIC inference over trusted SSH-forwarded TCP workers.

Workers bind loopback only. SHA-256 verifies checkpoint and response consistency;
it does not authenticate peers. Use SSH forwarding, never an untrusted network.
This is independent image-level inference, not synchronous distributed training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time

from PIL import Image
from ops import protocol

MAX_IMAGE = 4 * 1024 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_job(job: dict) -> None:
    if (set(job) != {"checkpoint_sha256", "color", "seed"}
            or not isinstance(job["checkpoint_sha256"], str)
            or len(job["checkpoint_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in job["checkpoint_sha256"])
            or job["color"] not in ("red", "blue")
            or type(job["seed"]) is not int or not 0 <= job["seed"] < 2**63):
        raise ValueError("invalid inference request")


def select_checkpoint_format(name, probe, warmstart):
    """Choose the validated loader and sampler for the explicit checkpoint format."""
    if name == "probe":
        return probe.load_checkpoint, probe.sample
    if name == "warmstart":
        return warmstart.load_warmstart_checkpoint, warmstart.sample
    raise ValueError("unknown checkpoint format")


def handle(sock: socket.socket, checkpoint_hash: str, sampler, runtime=None) -> None:
    """Handle a single request using an injected sampler for testability."""
    try:
        job = protocol.recv_json(sock)
        validate_job(job)
        if job["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("checkpoint hash mismatch")
        png = sampler(job["color"], job["seed"])
        if not isinstance(png, bytes) or not 0 < len(png) <= MAX_IMAGE:
            raise ValueError("invalid image payload")
        protocol.send_json(sock, {"status": "ok", "job": job, "sha256": digest(png),
                                  "bytes": len(png), "runtime": runtime or {}})
        sock.sendall(png)
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        protocol.send_json(sock, {"status": "error", "message": str(exc)})


def request(endpoint: tuple[str, int], job: dict, timeout: float) -> tuple[bytes, float, dict]:
    validate_job(job)
    start = time.monotonic()
    deadline = start + timeout
    with socket.create_connection(endpoint, timeout=protocol.remaining(deadline)) as sock:
        sock.settimeout(protocol.remaining(deadline))
        protocol.send_json(sock, job)
        header = protocol.recv_json(sock, deadline=deadline)
        if header.get("status") != "ok":
            raise ValueError(f"worker rejected request: {header.get('message', 'unknown')}")
        size = header.get("bytes")
        if header.get("job") != job or type(size) is not int or not 0 < size <= MAX_IMAGE:
            raise ValueError("malformed worker response")
        if not isinstance(header.get("runtime"), dict):
            raise ValueError("malformed worker runtime")
        data = protocol.recv_exact(sock, size, deadline=deadline)
        protocol.remaining(deadline)
    if digest(data) != header.get("sha256"):
        raise ValueError("image checksum mismatch")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (64, 64):
            raise ValueError("worker did not return a 64x64 PNG")
        image.verify()
    return data, time.monotonic() - start, header["runtime"]


def run(jobs: list[dict], endpoints: list[tuple[str, int]], output: Path,
        timeout: float = 600.0, *, resume: bool = False) -> list[dict]:
    """Run or resume a frozen plan; publish each image and record atomically."""
    if not endpoints or timeout <= 0 or not jobs:
        raise ValueError("workers, jobs and positive timeout are required")
    for job in jobs:
        validate_job(job)
    keys = [(j["color"], j["seed"]) for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate output jobs")
    plan = {"schema_version": 1, "jobs": jobs}
    if resume:
        if not output.is_dir() or output.is_symlink():
            raise ValueError("resume requires an existing output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / ".coordinator.lock"
    try:
        lock_file = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("coordinator lock exists; if no coordinator is running, remove it manually") from exc
    with lock_file:
        lock_file.write(str(os.getpid()) + "\n")
    try:
        plan_path = output / "plan.json"
        if resume:
            with plan_path.open(encoding="utf-8") as file:
                stored = json.load(file)
            if stored != plan:
                raise ValueError("resume job plan differs from frozen plan")
        else:
            with plan_path.open("x", encoding="utf-8") as file:
                json.dump(plan, file, indent=2, allow_nan=False)
                file.write("\n")

        def directory(job):
            return output / f"{job['color']}-{job['seed']}"

        def read_record(job):
            target = directory(job)
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f"invalid completed job directory: {target}")
            image_path, record_path = target / "image.png", target / "record.json"
            if image_path.is_symlink() or record_path.is_symlink():
                raise ValueError(f"unexpected symlink in {target}")
            with record_path.open(encoding="utf-8") as file:
                record = json.load(file)
            data = image_path.read_bytes()
            if (not isinstance(record, dict) or record.get("job") != job
                    or record.get("file") != f"{target.name}/image.png"
                    or record.get("sha256") != digest(data) or record.get("bytes") != len(data)
                    or not 0 < len(data) <= MAX_IMAGE):
                raise ValueError(f"completed job validation failed: {target}")
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or image.size != (64, 64):
                    raise ValueError(f"invalid completed PNG: {target}")
                image.verify()
            return record

        existing = {}
        for index, job in enumerate(jobs):
            target = directory(job)
            if target.exists() or target.is_symlink():
                existing[index] = read_record(job)

        def run_one(index_job):
            index, job = index_job
            failures = []
            for offset in range(len(endpoints)):
                endpoint = endpoints[(index + offset) % len(endpoints)]
                try:
                    data, elapsed, runtime = request(endpoint, job, timeout)
                except (OSError, EOFError, ValueError) as exc:
                    failures.append(str(exc))
                    continue
                target = directory(job)
                record = {"job": job, "file": f"{target.name}/image.png", "sha256": digest(data),
                          "bytes": len(data), "roundtrip_seconds": elapsed,
                          "worker_index": (index + offset) % len(endpoints),
                          "failed_attempts": len(failures), "runtime": runtime}
                staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output))
                try:
                    (staging / "image.png").write_bytes(data)
                    with (staging / "record.json").open("x", encoding="utf-8") as file:
                        json.dump(record, file, indent=2, allow_nan=False)
                        file.write("\n")
                    if target.exists() or target.is_symlink():
                        raise FileExistsError(f"job output already exists: {target}")
                    staging.rename(target)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
                return record
            raise RuntimeError(f"all workers failed for {job}: {failures}")

        start = time.monotonic()
        pending = [(i, job) for i, job in enumerate(jobs) if i not in existing]
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            completed = list(pool.map(run_one, pending))
        by_job = {(r["job"]["color"], r["job"]["seed"]): r
                  for r in [*existing.values(), *completed]}
        records = [by_job[(job["color"], job["seed"])] for job in jobs]
        manifest = {"schema_version": 1, "jobs": records, "workers": len(endpoints),
                    "segment_wall_seconds": time.monotonic() - start, "reused_jobs": len(existing),
                    "note": "Segment time excludes earlier attempts; image-level sharding. "
                            "Cross-device bitwise reproducibility is not guaranteed."}
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=output)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, allow_nan=False)
                file.write("\n")
            os.replace(temporary, output / "manifest.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return records
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--port", type=int, required=True)
    worker.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    worker.add_argument("--format", choices=("probe", "warmstart"), required=True)
    client = sub.add_parser("run")
    client.add_argument("--worker-port", type=int, action="append", required=True,
                        help="local SSH-forwarded port; repeat for each worker")
    client.add_argument("--checkpoint-sha256", required=True)
    client.add_argument("--seed", type=int, action="append", required=True)
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--resume", action="store_true", help="resume the exact frozen job plan")
    args = parser.parse_args()
    if args.command == "run":
        jobs = [{"checkpoint_sha256": args.checkpoint_sha256, "color": color, "seed": seed}
                for seed in args.seed for color in ("red", "blue")]
        records = run(jobs, [("127.0.0.1", port) for port in args.worker_port], args.output,
                      resume=args.resume)
        print(f"Completed {len(records)} samples; see {args.output / 'manifest.json'}")
        return
    import platform
    import PIL
    import torch
    from exposure import mosaic_pilotunet_probe as probe
    from exposure import mosaic_pilotunet_warmstart as warmstart
    from exposure import pilot_train
    loader, sample = select_checkpoint_format(args.format, probe, warmstart)
    checkpoint_data = args.checkpoint.read_bytes()
    checkpoint_hash = digest(checkpoint_data)
    checkpoint = loader(args.checkpoint)
    if digest(args.checkpoint.read_bytes()) != checkpoint_hash:
        raise ValueError("checkpoint changed during loading")
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "cuda_runtime": torch.version.cuda, "pillow": PIL.__version__,
               "device": args.device, "checkpoint_format": args.format,
               "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
               "source_sha256": {Path(module.__file__).name: digest(Path(module.__file__).read_bytes())
                                 for module in (probe, warmstart, pilot_train)},
               "worker_source_sha256": digest(Path(__file__).read_bytes())}

    def sampler(color, seed):
        image = sample(checkpoint, color, seed=seed, device=args.device)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
        listener.listen()
        print(f"worker listening on loopback:{args.port}; checkpoint sha256={checkpoint_hash}", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(600)
                try:
                    handle(connection, checkpoint_hash, sampler, runtime)
                except (OSError, EOFError):
                    continue


if __name__ == "__main__":
    main()
