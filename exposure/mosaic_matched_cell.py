"""Run the existing PilotUNet probe/warmstart on a selected MOSAIC cell.

A fresh interpreter is required for each cell. The wrapper records its source
hash in the underlying probe's runtime fingerprint. Exploratory only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from exposure import mosaic_pilotunet_probe as probe
from exposure import mosaic_pilotunet_warmstart as warmstart
from exposure.mosaic_toy_train import load_cells


def configure(cell):
    if cell not in {(a, b) for a in ('low', 'high') for b in ('balanced', 'imbalanced')}:
        raise ValueError('Invalid cell')
    probe.CELL = cell

    def load_cell(dataset):
        dataset = Path(dataset).resolve(strict=True)
        cells, manifest_hash = load_cells(dataset)
        items = cells[cell]
        images = torch.stack([item[0] for item in items])
        colors = ['red' if label == 0 else 'blue' for _, label in items]
        manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
        records = [r for r in manifest['records'] if r.get('split') == 'train'
                   and (r.get('complexity'), r.get('imbalance')) == cell]
        if len(records) != len(colors):
            raise ValueError('Training records do not match the loaded cell')
        provenance = []
        for record, color in zip(records, colors, strict=True):
            if record.get('target_color') != color:
                raise ValueError('Record order does not match loaded labels')
            provenance.append({'image': record['image'], 'target_color': color,
                               'pixel_sha256': record['pixel_sha256']})
        digest = hashlib.sha256()
        for record in provenance:
            digest.update(record['image'].encode('utf-8'))
            digest.update(record['pixel_sha256'].encode('utf-8'))
        return images, colors, manifest_hash, provenance, digest.hexdigest()

    probe.load_cell = load_cell
    original_runtime = probe._runtime

    def runtime(device):
        result = original_runtime(device)
        result['code_sha256']['mosaic_matched_cell'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    probe._runtime = runtime


def correct_report(module, cell):
    """Correct legacy low/balanced prose; the structured cell was already set."""
    original = module.run

    def run(*args, **kwargs):
        report = original(*args, **kwargs)
        report['limitations'] = [line.replace('Single cell (low/balanced)',
                                             f'Single cell ({cell[0]}/{cell[1]})')
                                 for line in report['limitations']]
        report['note'] = ('Matched-cell exploratory PilotUNet run; this individual cell '
                          'does not establish a MOSAIC contrast or Stage 0 passage.')
        path = Path(report['output']) / 'report.json'
        temporary = path.with_suffix('.corrected.tmp')
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write('\n')
        temporary.replace(path)
        return report

    module.run = run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complexity', required=True, choices=('low', 'high'))
    parser.add_argument('--imbalance', required=True, choices=('balanced', 'imbalanced'))
    parser.add_argument('--trainer', required=True, choices=('probe', 'warmstart'))
    args, rest = parser.parse_known_args()
    cell = (args.complexity, args.imbalance)
    configure(cell)
    module = probe if args.trainer == 'probe' else warmstart
    correct_report(module, cell)
    sys.argv = [sys.argv[0], *rest]
    module.main()


if __name__ == '__main__':
    main()
