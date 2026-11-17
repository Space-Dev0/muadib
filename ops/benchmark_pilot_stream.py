"""Tiny paired CPU pilot timing; no GPU, LAN, throughput or steady-state claim."""
import argparse
import json
import platform
from pathlib import Path
import torch
from exposure.pilot_train import train
from ops.validate_pilot_stream import ARGS, RULES, equal, streamed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(1)
    measurements = []
    for batch_size in (2, 8, 16):
        params = {**ARGS, 'batch_size': batch_size}
        baseline, a = train(RULES, None, steps=1, **params)
        remote, b = streamed(steps=1, batch_size=batch_size)
        equal(baseline, remote)
        measurements.append({'batch_size': batch_size,
                             'baseline_train_loop_seconds': a['elapsed_seconds'],
                             'stream_train_loop_seconds': b['elapsed_seconds'],
                             'stream_render_roundtrip_seconds': b['current_segment_remote_render_seconds'],
                             'stream_png_payload_bytes': b['current_segment_png_payload_bytes'],
                             'exact_model_ema_loss_counts_rng_parity': True})
    result = {'description': 'One CPU update per mode and batch; width 4, 4 timesteps, same host localhost TCP',
              'platform': platform.platform(), 'python': platform.python_version(),
              'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
              'measurements': measurements,
              'limitations': 'One trial per batch, no warmup; synchronous per-image calls. Round trip includes remote rendering, TCP socket wait and decode; no GPU idle, LAN bandwidth, steady state, or validated performance improvement.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
