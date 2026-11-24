"""Optional host-level Linux TCP retransmission counter.

This is an adjunct to, not a replacement for, ``ops.network_probe``. It invokes
an installed bpftrace without a shell or privilege escalation and fails closed
when the tracepoint, privileges, or collector are unavailable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import re
import shutil
import subprocess
import time

_BEGIN = "MUADIB_TCP_BEGIN"
_END = "MUADIB_TCP_END"
_COUNT = re.compile(r"^@retransmits:\s*(\d+)\s*$", re.MULTILINE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _program(port: int, seconds: int) -> str:
    # Values are validated integers, never executable input or shell text.
    return f'''BEGIN {{ printf("{_BEGIN}\\n"); }}
tracepoint:tcp:tcp_retransmit_skb /args->sport == {port} || args->dport == {port}/ {{
    @retransmits = count();
}}
interval:s:{seconds} {{ exit(); }}
END {{ printf("{_END}\\n"); }}'''


def collect_retransmissions(*, port: int, seconds: int = 5) -> dict:
    """Count host TCP retransmission events for sockets using ``port``.

    Filtering is host-wide by either endpoint's TCP port, not by process,
    peer, tunnel, experiment, or job. A successful zero is returned only after
    the collector starts and exits normally. In every failure case the count is
    null. This function is synchronous and does not start traffic itself.
    """
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 60:
        raise ValueError("seconds must be an integer between 1 and 60")

    result = {
        "available": False,
        "tcp_retransmit_events": None,
        "hook": "tracepoint:tcp:tcp_retransmit_skb",
        "platform": platform.system(),
        "port": port,
        "requested_interval_seconds": seconds,
        "scope": "all host TCP sockets with matching source or destination port",
        "attribution": "host port only; not an individual job, process, or VPN peer",
        "dropped_events": None,
        "dropped_events_note": "not reported by this collector; completeness is not established",
        "actual_trace_window": None,
        "actual_trace_window_note": "exact attachment and detachment times are not recorded",
        "collector_start_utc": None,
        "collector_end_utc": None,
        "collector_elapsed_seconds": None,
        "error": None,
    }
    if platform.system() != "Linux":
        result["error"] = "eBPF collection requires Linux"
        return result
    executable = shutil.which("bpftrace")
    if not executable:
        result["error"] = "bpftrace is not installed or not on PATH"
        return result

    result["collector_start_utc"] = _utc_now()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [executable, "-q", "-e", _program(port, seconds)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=seconds + 15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = f"collector failed: {type(exc).__name__}"
        return result
    finally:
        result["collector_end_utc"] = _utc_now()
        result["collector_elapsed_seconds"] = time.monotonic() - started
    if completed.returncode != 0:
        # Avoid including arbitrary environment data or sensitive process output.
        result["error"] = f"bpftrace exited with status {completed.returncode}; check permissions and tracepoint support"
        return result
    lines = completed.stdout.splitlines()
    counts = _COUNT.findall(completed.stdout)
    if lines.count(_BEGIN) != 1 or lines.count(_END) != 1 or len(counts) > 1:
        result["error"] = "collector output was incomplete or unexpected"
        return result
    # Unexpected output or warnings may signal loss or a failed hook. Do not
    # silently convert an incomplete capture into a successful zero.
    other = [line for line in lines if line.strip() and line not in (_BEGIN, _END)
             and not _COUNT.fullmatch(line)]
    if other or completed.stderr.strip():
        result["error"] = "collector reported warnings or unexpected output"
        return result
    result["tcp_retransmit_events"] = int(counts[0]) if counts else 0
    result["available"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True, help="TCP port to filter on this host")
    parser.add_argument("--seconds", type=int, default=5, help="trace duration, 1–60 seconds")
    args = parser.parse_args()
    try:
        result = collect_retransmissions(port=args.port, seconds=args.seconds)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
