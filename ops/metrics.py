"""Monotonic, application-level measurements; not GPU utilization or link speed."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time


@dataclass
class Metrics:
    scenes: int = 0
    payload_bytes: int = 0
    elapsed_seconds: float = 0.0
    receive_decode_seconds: float = 0.0

    def result(self) -> dict:
        data = asdict(self)
        data['scenes_per_second'] = self.scenes / self.elapsed_seconds if self.elapsed_seconds else None
        data['payload_mbps'] = self.payload_bytes * 8 / self.elapsed_seconds / 1e6 if self.elapsed_seconds else None
        return data


def append_jsonl(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
