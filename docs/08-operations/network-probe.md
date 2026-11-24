# SSH-forwarded network echo probe

`ops.network_probe` measures application TCP echo timings through a loopback
endpoint, commonly an SSH local forward. Launch the server on the remote host:

```sh
uv run python -m ops.network_probe serve --port 53220 --socket-timeout 5
```

Establish an authenticated SSH local forward, then probe its local port:

```sh
ssh -N -L 53220:127.0.0.1:53220 USER@HOST
uv run python -m ops.network_probe probe --port 53220 --count 20 \
  --interval 0.5 --timeout 3 --payload-size 1024 --output outputs/echo.json
```

The client `--timeout` is a whole-attempt deadline covering connect, send and
receive. The server `--socket-timeout` is a separate per-operation limit, not
negotiated or reported to the client. Record the server setting separately and
use a sufficiently generous value for the workload. A client-side failure can
have different causes; a server close is not proof of packet loss.

Each attempt uses fresh random payload bytes. The JSON records client settings,
individual successes and failures, a nearest-rank median and p95, and the
absolute differences between successive successful RTTs. This variation is
not one-way IP packet-delay variation. Echo RTT excludes connection setup but
includes application and SSH processing. It is not one-way delay.

The reported echoed-payload goodput divides useful echoed bytes by the entire
measurement interval, including failed attempts and intentional pacing. It is
not network link capacity, a GPU throughput measurement, or a speedup estimate.
An unpaced test still includes endpoint, connection setup and protocol overhead;
record payload size, interval, encryption, duration and repetitions before making
comparisons. This version changed the synthetic payload from zeros to fresh
random bytes; do not treat its goodput as directly comparable to the earlier
zero-payload smoke-test records. Existing JSON results are not rewritten.
