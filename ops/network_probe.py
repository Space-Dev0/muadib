"""Measure loopback/SSH-forwarded TCP echo latency and application goodput.

Run ``serve`` on a trusted worker and forward its loopback port through SSH.
The probe deliberately binds and connects only to loopback. Results are
application-level echo measurements, not one-way delay or raw link capacity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import socketserver
import struct
import time

_HEADER = struct.Struct("!QI")  # probe number, payload length
_MAX_PAYLOAD = 1024 * 1024


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("probe deadline exceeded")
    return remaining


def _read_exact(sock: socket.socket, count: int, deadline: float | None = None) -> bytes:
    data = bytearray()
    while len(data) < count:
        if deadline is not None:
            sock.settimeout(_remaining(deadline))
        part = sock.recv(count - len(data))
        if not part:
            raise EOFError("connection closed before complete echo")
        data.extend(part)
    return bytes(data)


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(self.server.socket_timeout)
        try:
            header = _read_exact(self.request, _HEADER.size)
            _, length = _HEADER.unpack(header)
            if not 1 <= length <= _MAX_PAYLOAD:
                return
            payload = _read_exact(self.request, length)
            self.request.sendall(header + payload)
        except (OSError, EOFError):
            return


class EchoServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, *, socket_timeout: float = 5.0):
        if not math.isfinite(socket_timeout) or not 0 < socket_timeout <= 3600:
            raise ValueError("server socket timeout must be finite and between 0 and 3600 seconds")
        self.socket_timeout = socket_timeout
        super().__init__(("127.0.0.1", port), _EchoHandler)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def probe(port: int, *, count: int = 20, interval: float = 0.5,
          timeout: float = 3.0, payload_size: int = 1024) -> dict:
    """Measure one forwarded loopback endpoint; every attempt is retained.

    Echo RTT starts after TCP connect and includes application/forwarding work.
    ``rtt_variation`` is absolute consecutive RTT difference, not RFC IPDV.
    Timeout is a monotonic per-attempt deadline including connect, send, and read.
    Payload goodput counts only echoed bytes of successful probes over the full
    measured window, including failures and the configured probe spacing.
    """
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not 1 <= count <= 10000:
        raise ValueError("count must be between 1 and 10000")
    if not math.isfinite(interval) or not 0 <= interval <= 3600:
        raise ValueError("interval must be finite and between 0 and 3600 seconds")
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("timeout must be finite and between 0 and 3600 seconds")
    if not 1 <= payload_size <= _MAX_PAYLOAD:
        raise ValueError(f"payload size must be between 1 and {_MAX_PAYLOAD} bytes")

    started = time.monotonic()
    previous_rtt = None
    attempts = []
    variations = []
    for index in range(count):
        if index:
            time.sleep(interval)
        # Fresh random bytes on every attempt avoid treating repeated zeros (or
        # a repeated random block) as representative of incompressible payloads.
        payload = os.urandom(payload_size)
        attempt_start = time.monotonic()
        deadline = attempt_start + timeout
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=_remaining(deadline)) as sock:
                sock.settimeout(_remaining(deadline))
                connect_ms = (time.monotonic() - attempt_start) * 1000
                expected = _HEADER.pack(index, payload_size) + payload
                echo_start = time.monotonic()
                sock.settimeout(_remaining(deadline))
                sock.sendall(expected)
                echoed = _read_exact(sock, len(expected), deadline)
                _remaining(deadline)
                rtt_ms = (time.monotonic() - echo_start) * 1000
            if echoed != expected:
                raise ValueError("echo contents did not match the request")
            attempts.append({"sequence": index, "status": "ok", "connect_ms": connect_ms,
                             "echo_rtt_ms": rtt_ms})
            if previous_rtt is not None:
                variations.append(abs(rtt_ms - previous_rtt))
            previous_rtt = rtt_ms
        except (OSError, EOFError, ValueError) as exc:
            attempts.append({"sequence": index, "status": "error", "error": type(exc).__name__,
                             "message": str(exc)})
            previous_rtt = None  # do not pretend probes separated by failure are adjacent
    elapsed = time.monotonic() - started
    rtts = [a["echo_rtt_ms"] for a in attempts if a["status"] == "ok"]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": {"host": "127.0.0.1", "port": port},
        "configuration": {"count": count, "interval_seconds": interval,
                          "timeout_seconds": timeout, "payload_bytes_each_direction": payload_size,
                          "payload_pattern": "fresh_random_bytes_per_attempt",
                          "server_socket_timeout_seconds": None},
        "summary": {"attempts": count, "successes": len(rtts), "failed_attempts": count - len(rtts),
                    "failure_fraction": (count - len(rtts)) / count,
                    "elapsed_seconds": elapsed, "echo_rtt_median_ms": _percentile(rtts, 0.5),
                    "echo_rtt_p95_ms": _percentile(rtts, 0.95),
                    "rtt_variation_count": len(variations),
                    "rtt_variation_median_ms": _percentile(variations, 0.5),
                    "rtt_variation_p95_ms": _percentile(variations, 0.95),
                    "echoed_payload_goodput_mbps": (len(rtts) * payload_size * 8 / elapsed / 1e6
                                                     if elapsed > 0 else None)},
        "notes": ["One new TCP connection per attempt; echo RTT excludes connection setup.",
                  "Application echo includes remote processing and SSH forwarding; not one-way delay.",
                  "RTT variation uses adjacent successful probes only; it is not RFC one-way IPDV.",
                  "Percentiles use the nearest-rank method, including the median.",
                  "Server socket timeout is not negotiated; configure and record it separately on the server.",
                  "Goodput counts echoed payload over the full window, including idle intervals; not link capacity."],
        "attempts": attempts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve", help="serve an echo on loopback; connect through an SSH tunnel")
    server.add_argument("--port", required=True, type=int)
    server.add_argument("--socket-timeout", type=float, default=5.0,
                        help="server per-operation socket timeout in seconds (default: 5)")
    client = sub.add_parser("probe", help="probe a loopback or SSH-forwarded endpoint")
    client.add_argument("--port", required=True, type=int)
    client.add_argument("--count", type=int, default=20)
    client.add_argument("--interval", type=float, default=0.5)
    client.add_argument("--timeout", type=float, default=3.0)
    client.add_argument("--payload-size", type=int, default=1024)
    client.add_argument("--output", type=Path, required=True, help="new JSON output path")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.command == "serve":
        try:
            with EchoServer(args.port, socket_timeout=args.socket_timeout) as listener:
                print(f"Echo server on 127.0.0.1:{args.port}; socket timeout {args.socket_timeout}s; use SSH forwarding for remote access.", flush=True)
                listener.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Could not start echo server: {exc}\n")
        return 0
    created_identity = None
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            stat = os.fstat(output.fileno())
            created_identity = (stat.st_dev, stat.st_ino)
            result = probe(args.port, count=args.count, interval=args.interval,
                           timeout=args.timeout, payload_size=args.payload_size)
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as exc:
        if created_identity is not None:
            try:
                stat = args.output.lstat()
                if (stat.st_dev, stat.st_ino) == created_identity:
                    args.output.unlink()
            except OSError:
                pass
        parser.exit(1, f"Probe failed: {exc}\n")
    print(f"Saved {result['summary']['successes']}/{args.count} successful probes to {args.output}")
    return 0 if result["summary"]["successes"] == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
