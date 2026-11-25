"""Run a frozen diffusion job plan through loopback or SSH-forwarded workers.

This coordinator reuses the existing worker protocol. It never opens a worker
on a public address, changes model settings, or drops incomplete planned jobs.
Results are image-level independent inference, not distributed model training.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import uuid

from PIL import Image
from ops import distributed_inference

_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_MAX_PROBE_BYTES = 1024 * 1024
_MAX_RESULT_RECORD_BYTES = 1024 * 1024


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must contain only letters, digits, underscores or hyphens")
    return value


def validate_plan(plan: object) -> list[dict]:
    """Validate the frozen logical job set; worker requests remain legacy compatible."""
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "jobs"} or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("plan must contain schema_version=1 and jobs")
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan jobs must be a nonempty list")
    ids, requests = set(), set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {"id", "stratum", "checkpoint_sha256", "color", "seed"}:
            raise ValueError("each job must contain id, stratum, checkpoint_sha256, color and seed")
        _identifier(job["id"], "job id")
        _identifier(job["stratum"], "stratum")
        wire = {key: job[key] for key in ("checkpoint_sha256", "color", "seed")}
        distributed_inference.validate_job(wire)
        signature = (job["checkpoint_sha256"], job["color"], job["seed"])
        if job["id"].casefold() in ids or signature in requests:
            raise ValueError("duplicate job id or checkpoint/color/seed")
        ids.add(job["id"].casefold())
        requests.add(signature)
    return jobs


def validate_workers(workers: object) -> list[dict]:
    if not isinstance(workers, list) or not workers:
        raise ValueError("workers must be a nonempty list")
    ids, ports = set(), set()
    for worker in workers:
        if not isinstance(worker, dict) or not {"id", "port", "checkpoint_sha256"} <= worker.keys():
            raise ValueError("worker requires id, port and checkpoint_sha256")
        if set(worker) - {"id", "port", "checkpoint_sha256", "estimate_seconds", "probe_file", "probe_port"}:
            raise ValueError("unknown worker setting")
        _identifier(worker["id"], "worker id")
        port = worker["port"]
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("worker port must be between 1 and 65535")
        distributed_inference.validate_job({"checkpoint_sha256": worker["checkpoint_sha256"],
                                            "color": "red", "seed": 0})
        estimate = worker.get("estimate_seconds", 60.0)
        if type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate < 0:
            raise ValueError("estimate_seconds must be finite and non-negative")
        if ("probe_file" in worker) != ("probe_port" in worker):
            raise ValueError("probe_file and probe_port must be specified together")
        if "probe_file" in worker:
            if not isinstance(worker["probe_file"], str) or not worker["probe_file"]:
                raise ValueError("probe_file must be a nonempty path")
            if type(worker["probe_port"]) is not int or not 1 <= worker["probe_port"] <= 65535:
                raise ValueError("probe_port must be between 1 and 65535")
        if worker["id"] in ids or port in ports:
            raise ValueError("duplicate worker id or port")
        ids.add(worker["id"])
        ports.add(port)
    return workers


def validate_assignments(assignments: object, jobs: list[dict], workers: list[dict]) -> dict[str, str]:
    """Validate a complete preferred-worker map; absent workers remain unavailable."""
    if (not isinstance(assignments, dict) or set(assignments) != {"schema_version", "assignments"}
            or type(assignments["schema_version"]) is not int or assignments["schema_version"] != 1
            or not isinstance(assignments["assignments"], dict)):
        raise ValueError("assignments must contain schema_version=1 and an assignments object")
    mapping = assignments["assignments"]
    if set(mapping) != {j["id"] for j in jobs}:
        raise ValueError("assignments must cover exactly the frozen job IDs")
    by_id = {w["id"]: w for w in workers}
    for job in jobs:
        preferred = _identifier(mapping[job["id"]], "preferred worker id")
        if preferred in by_id and by_id[preferred]["checkpoint_sha256"] != job["checkpoint_sha256"]:
            raise ValueError(f"preferred worker has an incompatible checkpoint for {job['id']}")
    return mapping


def _fresh_rtt(worker: dict, max_age: float, now: float | None = None) -> float | None:
    """Read pre-measured echo RTT, returning None for stale, invalid or failed probes."""
    if "probe_file" not in worker:
        return None
    try:
        path = worker["probe_file"]
        # Do not block on named pipes or follow links to devices.  Check both
        # before and after open to narrow path replacement races.
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_PROBE_BYTES:
            return None
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as f:
            current = os.fstat(f.fileno())
            if (not stat.S_ISREG(current.st_mode) or current.st_size > _MAX_PROBE_BYTES
                    or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)):
                return None
            raw = f.read(_MAX_PROBE_BYTES + 1)
        if len(raw) > _MAX_PROBE_BYTES:
            return None
        data = json.loads(raw)
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            return None
        config = data["configuration"]
        count, payload = config["count"], config["payload_bytes_each_direction"]
        interval, probe_timeout = config["interval_seconds"], config["timeout_seconds"]
        if (type(count) is not int or not 5 <= count <= 10000
                or type(payload) is not int or not 1 <= payload <= 1024 * 1024
                or type(interval) not in (int, float) or not math.isfinite(interval) or not 0 <= interval <= 3600
                or type(probe_timeout) not in (int, float) or not math.isfinite(probe_timeout)
                or not 0 < probe_timeout <= 3600):
            return None
        observed = datetime.fromisoformat(data["created_utc"].replace("Z", "+00:00"))
        if observed.tzinfo is None:
            return None
        age = (time.time() if now is None else now) - observed.timestamp()
        if age < -5 or age > max_age:
            return None
        endpoint = data["endpoint"]
        if endpoint != {"host": "127.0.0.1", "port": worker["probe_port"]}:
            return None
        summary = data["summary"]
        value = summary["echo_rtt_median_ms"]
        if (type(summary["successes"]) is not int or summary["successes"] != count
                or type(summary["attempts"]) is not int
                or summary["attempts"] != summary["successes"]
                or type(summary["failed_attempts"]) is not int
                or summary["failed_attempts"] != 0):
            return None
        attempts = data["attempts"]
        if not isinstance(attempts, list) or len(attempts) != count:
            return None
        rtts = []
        for index, attempt in enumerate(attempts):
            if (not isinstance(attempt, dict) or type(attempt.get("sequence")) is not int
                    or attempt["sequence"] != index or attempt.get("status") != "ok"):
                return None
            measurement = attempt.get("echo_rtt_ms")
            if type(measurement) not in (int, float) or not math.isfinite(measurement) or measurement < 0:
                return None
            rtts.append(measurement)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return None
        if value != sorted(rtts)[math.ceil(count * .5) - 1]:
            return None
        return float(value)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None


def _event(path: Path, event: dict) -> None:
    """Append an event before/after an attempt; preserve it across normal restarts."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _atomic_json(path: Path, data: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".staging-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _read_events_recover_tail(path: Path) -> list[dict]:
    """Preserve and repair only an incomplete final journal record.

    A complete malformed line is corruption and must not be discarded. A torn
    suffix is archived byte-for-byte before replacing the working journal.
    Recovery uncertainty remains visible in the new journal until all missing
    jobs have been resolved or their retry is explicitly acknowledged.
    """
    records = []
    valid_end = 0
    repair_at = None
    with path.open("rb") as source:
        while line := source.readline():
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                # A malformed terminated line, or any line followed by data,
                # is not demonstrably a crash-torn suffix.
                if line.endswith(b"\n") or source.read(1):
                    raise ValueError("journal contains malformed complete or middle line") from exc
                repair_at = valid_end
                break
            if not isinstance(event, dict):
                raise ValueError("journal event must be an object")
            records.append(event)
            valid_end = source.tell()
            if not line.endswith(b"\n"):
                # A valid but unterminated event must be separated before append.
                repair_at = valid_end
                break
    if repair_at is None:
        return records

    backup = path.with_name(f"events.torn-{uuid.uuid4().hex}.jsonl")
    with path.open("rb") as source, backup.open("xb") as archived:
        shutil.copyfileobj(source, archived)
        archived.flush()
        os.fsync(archived.fileno())
    recovery = {"event": "journal_recovered", "time_utc": _utc(),
                "backup": backup.name, "uncertain": True,
                "reason": "unterminated final record; inspect archive before retrying missing jobs"}
    fd, name = tempfile.mkstemp(prefix=".events-recovered-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as recovered, backup.open("rb") as archived:
            remaining = repair_at
            while remaining:
                block = archived.read(min(65536, remaining))
                if not block:
                    raise OSError("archived journal shortened during recovery")
                recovered.write(block)
                remaining -= len(block)
            if records and repair_at and not line.endswith(b"\n") and repair_at == valid_end:
                # Only append a newline if the preserved final record was valid.
                with backup.open("rb") as tail:
                    tail.seek(repair_at - 1)
                    if tail.read(1) != b"\n":
                        recovered.write(b"\n")
            recovered.write((json.dumps(recovery, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            recovered.flush()
            os.fsync(recovered.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    records.append(recovery)
    return records


def _verify_png(data: bytes) -> None:
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (64, 64):
            raise ValueError("invalid PNG payload")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()


def _load_valid_result(output: Path, job: dict) -> dict | None:
    target = output / "results" / job["id"]
    if not target.exists() and not target.is_symlink():
        return None
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f"invalid result directory for {job['id']}")
    img, record_file = target / "image.png", target / "record.json"
    if img.is_symlink() or record_file.is_symlink():
        raise ValueError(f"symlink in result for {job['id']}")
    try:
        before = record_file.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_RESULT_RECORD_BYTES:
            raise ValueError("invalid result record size or file type")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(record_file, flags)
        with os.fdopen(fd, "rb") as file:
            current = os.fstat(file.fileno())
            if (not stat.S_ISREG(current.st_mode) or not 0 < current.st_size <= _MAX_RESULT_RECORD_BYTES
                    or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)):
                raise ValueError("invalid result record size or file type")
            raw = file.read(_MAX_RESULT_RECORD_BYTES + 1)
        if len(raw) > _MAX_RESULT_RECORD_BYTES:
            raise ValueError("invalid result record size")
        record = json.loads(raw)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"result validation failed for {job['id']}: invalid record") from exc
    if not 0 < img.stat().st_size <= distributed_inference.MAX_IMAGE:
        raise ValueError(f"result validation failed for {job['id']}: invalid image size")
    with img.open("rb") as source:
        data = source.read(distributed_inference.MAX_IMAGE + 1)
    if (not isinstance(record, dict) or record.get("job") != job or record.get("sha256") != hashlib.sha256(data).hexdigest()
            or record.get("bytes") != len(data) or not 0 < len(data) <= distributed_inference.MAX_IMAGE):
        raise ValueError(f"result validation failed for {job['id']}")
    try:
        _verify_png(data)
    except (OSError, ValueError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError(f"invalid PNG for {job['id']}: {type(exc).__name__}") from exc
    return record


def _publish(output: Path, job: dict, data: bytes, worker: dict, elapsed: float,
             runtime: dict, attempt_id: str) -> dict:
    if not isinstance(data, bytes) or not 0 < len(data) <= distributed_inference.MAX_IMAGE:
        raise ValueError("invalid PNG payload")
    # Check framing and decode pixel data; verify() alone does not decode IDAT.
    _verify_png(data)
    target = output / "results" / job["id"]
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output / "results"))
    record = {"job": job, "worker_id": worker["id"], "attempt_id": attempt_id,
              "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
              "roundtrip_seconds": elapsed, "runtime": runtime, "completed_utc": _utc()}
    try:
        with (staging / "image.png").open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with (staging / "record.json").open("x", encoding="utf-8") as f:
            json.dump(record, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"result already exists: {target}")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return record


def run(plan: dict, workers: list[dict], output: Path, *, policy: str = "round-robin",
        timeout: float = 600, telemetry_max_age: float = 15, resume: bool = False,
        allow_uncertain_retry: bool = False, assignments: dict | None = None,
        allow_assignment_failover: bool = False) -> dict:
    """Run one attempt per compatible worker per job; retain partial runs for resume.

    The adaptive heuristic uses configured total-time priors or an EWMA of observed
    request wall times plus fresh optional echo RTT as a conservative routing score.
    It is not a prediction guarantee; echo RTT is not raw network latency.
    """
    setup_start = time.monotonic()
    jobs = validate_plan(plan)
    workers = validate_workers(workers)
    preferred_workers = validate_assignments(assignments, jobs, workers) if assignments is not None else None
    if allow_assignment_failover and preferred_workers is None:
        raise ValueError("assignment failover requires a preferred-worker map")
    if policy not in ("round-robin", "adaptive"):
        raise ValueError("policy must be round-robin or adaptive")
    if (not math.isfinite(timeout) or timeout <= 0 or
            not math.isfinite(telemetry_max_age) or telemetry_max_age <= 0):
        raise ValueError("timeout and telemetry_max_age must be finite and positive")
    if preferred_workers is None and any(not any(w["checkpoint_sha256"] == j["checkpoint_sha256"] for w in workers) for j in jobs):
        raise ValueError("every job must have at least one compatible checkpoint worker")
    output = Path(output)
    if resume:
        if output.is_symlink() or not output.is_dir():
            raise ValueError("resume requires an existing non-symlink output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / ".coordinator.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("coordinator lock exists; check process before removing a stale lock") from exc
    lock_stat = os.fstat(fd)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        plan_file = output / "plan.json"
        for name in ("plan.json", "assignments.json", "events.jsonl", "summary.json"):
            if (output / name).is_symlink():
                raise ValueError(f"symlink is not allowed: {name}")
        if resume:
            with plan_file.open(encoding="utf-8") as f:
                if json.load(f) != plan:
                    raise ValueError("resume plan differs from frozen plan")
        else:
            _atomic_json(plan_file, plan)
        assignment_file = output / "assignments.json"
        if resume:
            if assignment_file.exists() != (assignments is not None):
                raise ValueError("resume assignments differ from frozen assignments")
            if assignment_file.exists():
                with assignment_file.open(encoding="utf-8") as f:
                    if json.load(f) != assignments:
                        raise ValueError("resume assignments differ from frozen assignments")
        elif assignments is not None:
            _atomic_json(assignment_file, assignments)
        results = output / "results"
        results.mkdir(exist_ok=True)
        if results.is_symlink() or not results.is_dir():
            raise ValueError("invalid results directory")
        events = output / "events.jsonl"
        validation_start = time.monotonic()
        existing = {j["id"]: result for j in jobs if (result := _load_valid_result(output, j)) is not None}
        result_validation_seconds = time.monotonic() - validation_start
        initial_reused = len(existing)
        # Recover dangling starts left by an interrupted coordinator; do not infer outcome.
        if resume and events.exists():
            started, ended, uncertain_jobs = {}, set(), set()
            for event in _read_events_recover_tail(events):
                if event.get("event") == "attempt_started":
                    started[event["attempt_id"]] = event
                elif event.get("event") in ("attempt_succeeded", "attempt_failed", "attempt_abandoned"):
                    ended.add(event["attempt_id"])
                    if event.get("uncertain") or event.get("event") == "attempt_abandoned":
                        uncertain_jobs.add(event["job_id"])
                elif event.get("event") == "journal_recovered":
                    uncertain_jobs.update(j["id"] for j in jobs if j["id"] not in existing)
            for attempt_id in started.keys() - ended:
                uncertain_jobs.add(started[attempt_id]["job_id"])
                _event(events, {"event": "attempt_abandoned", "attempt_id": attempt_id,
                                "session_id": started[attempt_id].get("session_id"),
                                "worker_id": started[attempt_id].get("worker_id"),
                                "stratum": started[attempt_id].get("stratum"),
                                "job_id": started[attempt_id]["job_id"], "time_utc": _utc(),
                                "uncertain": True,
                                "reason": "coordinator stopped before recording an outcome"})
            unresolved = uncertain_jobs - existing.keys()
            if unresolved and not allow_uncertain_retry:
                raise RuntimeError("Uncertain prior attempts may still run remotely; inspect workers, then "
                                   "resume with --allow-uncertain-retry: " + ", ".join(sorted(unresolved)))
        session_id = uuid.uuid4().hex
        try:
            coordinator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        except OSError:
            coordinator_sha256 = None
        # Never store SSH destinations, credentials, or filesystem probe paths.
        session_config = {
            "schema_version": 1, "coordinator_source_sha256": coordinator_sha256,
            "policy": policy, "timeout_seconds": timeout, "telemetry_max_age_seconds": telemetry_max_age,
            "resume": resume, "allow_uncertain_retry": allow_uncertain_retry,
            "assignment_controlled": preferred_workers is not None,
            "allow_assignment_failover": allow_assignment_failover,
            "workers": [{"id": w["id"], "port": w["port"], "checkpoint_sha256": w["checkpoint_sha256"],
                         "estimate_seconds": w.get("estimate_seconds", 60.0),
                         "probe_enabled": "probe_file" in w, "probe_port": w.get("probe_port")}
                        for w in workers],
        }
        _event(events, {"event": "run_started", "session_id": session_id, "time_utc": _utc(),
                        "policy": policy, "reused_jobs": len(existing), "session_config": session_config})
        pending = deque((i, job) for i, job in enumerate(jobs) if job["id"] not in existing)
        tried = {j["id"]: set() for _, j in pending}
        failed = set()
        busy = set()
        quarantined = set()
        unavailable = set()
        futures = {}
        history = {}
        start = time.monotonic()
        setup_seconds = start - setup_start
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            while pending or futures:
                # Rotate pending work so an unavailable checkpoint cannot block other jobs.
                for _ in range(len(pending)):
                    index, job = pending.popleft()
                    preferred = preferred_workers[job["id"]] if preferred_workers is not None else None
                    if preferred is not None:
                        preferred_entry = next((w for w in workers if w["id"] == preferred), None)
                        preferred_unavailable = (preferred_entry is None or preferred in unavailable
                                                 or preferred in quarantined or preferred in tried[job["id"]])
                        if not preferred_unavailable:
                            if preferred in busy:
                                pending.append((index, job))
                                continue
                            compatible = [preferred_entry]
                        elif allow_assignment_failover:
                            compatible = [w for w in workers if w["checkpoint_sha256"] == job["checkpoint_sha256"]
                                          and w["id"] not in tried[job["id"]]
                                          and w["id"] not in quarantined and w["id"] not in unavailable]
                        else:
                            failed.add(job["id"])
                            _event(events, {"event": "assignment_unavailable", "session_id": session_id,
                                            "job_id": job["id"], "stratum": job["stratum"],
                                            "preferred_worker_id": preferred, "time_utc": _utc()})
                            continue
                    else:
                        compatible = [w for w in workers if w["checkpoint_sha256"] == job["checkpoint_sha256"]
                                      and w["id"] not in tried[job["id"]] and w["id"] not in quarantined]
                    if not compatible:
                        failed.add(job["id"])
                        continue
                    candidates = [w for w in compatible if w["id"] not in busy]
                    if not candidates:
                        pending.append((index, job))
                        continue
                    scores = {}
                    if policy == "adaptive":
                        for w in candidates:
                            rtt = _fresh_rtt(w, telemetry_max_age)
                            base = history.get(w["id"], float(w.get("estimate_seconds", 60.0)))
                            scores[w["id"]] = (base + (rtt / 1000 if rtt is not None else 0), rtt)
                        worker = min(candidates, key=lambda w: (scores[w["id"]][0], workers.index(w)))
                    else:
                        worker = min(candidates, key=lambda w: (workers.index(w) - index) % len(workers))
                    attempt_id = uuid.uuid4().hex
                    if preferred is not None and worker["id"] != preferred:
                        reason = ("absent" if preferred_entry is None else
                                  "quarantined" if preferred in quarantined else "connection_refused")
                        _event(events, {"event": "assignment_deviation", "session_id": session_id,
                                        "job_id": job["id"], "stratum": job["stratum"],
                                        "preferred_worker_id": preferred, "actual_worker_id": worker["id"],
                                        "reason": reason, "time_utc": _utc()})
                    rtt = scores.get(worker["id"], (None, None))[1]
                    _event(events, {"event": "attempt_started", "session_id": session_id,
                                    "attempt_id": attempt_id, "job_id": job["id"],
                                    "stratum": job["stratum"], "worker_id": worker["id"],
                                    "time_utc": _utc(), "probe_rtt_ms": rtt,
                                    "routing_score_seconds": scores.get(worker["id"], (None, None))[0]})
                    tried[job["id"]].add(worker["id"])
                    busy.add(worker["id"])
                    wire = {key: job[key] for key in ("checkpoint_sha256", "color", "seed")}
                    future = pool.submit(distributed_inference.request, ("127.0.0.1", worker["port"]), wire, timeout)
                    futures[future] = (index, job, worker, attempt_id, time.monotonic())
                if not futures:
                    if pending:
                        raise RuntimeError("scheduler made no progress")
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index, job, worker, attempt_id, attempt_start = futures.pop(future)
                    busy.remove(worker["id"])
                    elapsed = time.monotonic() - attempt_start
                    try:
                        data, reported_elapsed, runtime = future.result()
                        if not isinstance(runtime, dict) or not math.isfinite(reported_elapsed) or reported_elapsed < 0:
                            raise ValueError("invalid worker timing or runtime")
                    except (OSError, EOFError, ValueError, TypeError, RuntimeError, Image.DecompressionBombError, SyntaxError) as exc:
                        # A refused connection cannot have submitted work; other errors
                        # can occur after submission and do not prove cancellation.
                        uncertain = not isinstance(exc, ConnectionRefusedError)
                        if uncertain:
                            quarantined.add(worker["id"])
                            failed.add(job["id"])
                        elif preferred_workers is not None:
                            unavailable.add(worker["id"])
                        _event(events, {"event": "attempt_failed", "session_id": session_id,
                                        "attempt_id": attempt_id, "job_id": job["id"],
                                        "stratum": job["stratum"], "worker_id": worker["id"],
                                        "time_utc": _utc(), "duration_seconds": elapsed,
                                        "error_type": type(exc).__name__, "error": str(exc)[:500],
                                        "uncertain": uncertain})
                        if not uncertain:
                            pending.append((index, job))
                        continue
                    try:
                        record = _publish(output, job, data, worker, float(reported_elapsed), runtime, attempt_id)
                    except (OSError, ValueError, TypeError, RuntimeError, Image.DecompressionBombError, SyntaxError) as exc:
                        # The request returned, so the remote operation is over.
                        # Do not retry a failed local write during this session:
                        # the filesystem or payload may need operator attention.
                        failed.add(job["id"])
                        _event(events, {"event": "attempt_failed", "session_id": session_id,
                                        "attempt_id": attempt_id, "job_id": job["id"],
                                        "stratum": job["stratum"], "worker_id": worker["id"],
                                        "time_utc": _utc(), "duration_seconds": elapsed,
                                        "error_type": type(exc).__name__, "error": str(exc)[:500],
                                        "uncertain": False, "failure_phase": "publication"})
                        continue
                    history[worker["id"]] = (0.7 * history[worker["id"]] + 0.3 * reported_elapsed
                                               if worker["id"] in history else reported_elapsed)
                    existing[job["id"]] = record
                    _event(events, {"event": "attempt_succeeded", "session_id": session_id,
                                    "attempt_id": attempt_id, "job_id": job["id"],
                                    "stratum": job["stratum"], "worker_id": worker["id"],
                                    "time_utc": _utc(), "duration_seconds": elapsed,
                                    "roundtrip_seconds": float(reported_elapsed), "bytes": len(data)})
        planned = Counter(j["stratum"] for j in jobs)
        completed = Counter(j["stratum"] for j in jobs if j["id"] in existing)
        duration = time.monotonic() - start
        summary = {"schema_version": 1, "session_id": session_id, "policy": policy,
                   "planned": len(jobs), "completed": len(existing), "missing": len(jobs) - len(existing),
                   "failed_this_session": sorted(failed), "segment_wall_seconds": duration,
                   "setup_seconds": setup_seconds, "result_validation_seconds": result_validation_seconds,
                   "session_elapsed_before_summary_seconds": time.monotonic() - setup_start,
                   "reused_jobs": initial_reused,
                   "strata": {s: {"planned": count, "completed": completed[s],
                                  "missing": count - completed[s]} for s, count in planned.items()},
                   "note": "Segment time excludes previous sessions; unfinished jobs are retained. "
                           "Cross-device bitwise reproducibility is not guaranteed."}
        _atomic_json(output / "summary.json", summary)
        _event(events, {"event": "run_finished", "session_id": session_id, "time_utc": _utc(),
                        "completed": len(existing), "missing": len(jobs) - len(existing),
                        "duration_seconds": duration})
        return summary
    finally:
        try:
            current = lock.lstat()
            if (current.st_dev, current.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
                lock.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="immutable versioned job-plan JSON")
    parser.add_argument("--workers", type=Path, required=True, help="loopback-forwarded worker configuration JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("round-robin", "adaptive"), default="round-robin")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--telemetry-max-age", type=float, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-uncertain-retry", action="store_true",
                        help="acknowledge that unfinished attempts may still run remotely")
    parser.add_argument("--assignments", type=Path, help="freeze an exact job-to-preferred-worker map")
    parser.add_argument("--allow-assignment-failover", action="store_true",
                        help="allow recorded deviations only when preferred worker is unavailable")
    args = parser.parse_args(argv)
    try:
        with args.plan.open(encoding="utf-8") as f:
            plan = json.load(f)
        with args.workers.open(encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict) or set(config) != {"workers"}:
            raise ValueError("worker configuration must contain only workers")
        assignments = json.loads(args.assignments.read_text(encoding="utf-8")) if args.assignments else None
        summary = run(plan, config["workers"], args.output, policy=args.policy,
                      timeout=args.timeout, telemetry_max_age=args.telemetry_max_age, resume=args.resume,
                      allow_uncertain_retry=args.allow_uncertain_retry, assignments=assignments,
                      allow_assignment_failover=args.allow_assignment_failover)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["missing"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
