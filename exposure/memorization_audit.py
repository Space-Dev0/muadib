"""Descriptive nearest-training-image audit for the fixed MOSAIC pilot.

Raw RGB MSE is not a memorization detector. Results can reflect background,
noise, blur, position, and different source distributions. Controls are paired
by target color and complexity, not independent model-training replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CELLS = ('low-balanced', 'low-imbalanced', 'high-balanced', 'high-imbalanced')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rgb(path: Path, *, expected_pixel: str | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (64, 64): raise ValueError('expected 64x64 image')
        pixels = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    if expected_pixel and hashlib.sha256(pixels.tobytes()).hexdigest() != expected_pixel:
        raise ValueError(f'pixel hash mismatch: {path.name}')
    return pixels.astype(np.float32) / 255.0


def nearest_mse(query: np.ndarray, references: np.ndarray) -> tuple[int, float]:
    """Nearest raw RGB mean-square error on [0,1], deterministic first tie."""
    query = np.asarray(query, dtype=np.float32)
    references = np.asarray(references, dtype=np.float32)
    if (query.shape != (64,64,3) or references.ndim != 4 or references.shape[1:] != query.shape
            or not len(references) or not np.isfinite(query).all() or not np.isfinite(references).all()
            or query.min() < 0 or query.max() > 1 or references.min() < 0 or references.max() > 1):
        raise ValueError('expected finite normalized RGB images and nonempty references')
    distances = np.mean(np.square(references - query), axis=(1,2,3), dtype=np.float64)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _summary(rows):
    values=np.asarray([row['mse'] for row in rows],dtype=float)
    return {'count':len(values), 'median_mse':float(np.median(values)),
            'min_mse':float(values.min()), 'max_mse':float(values.max()),
            'exact_pixel_matches':int(np.count_nonzero(values==0.0))}


def audit(dataset: Path, evaluation: Path, *, expected_manifest_sha256: str) -> dict:
    manifest_path=dataset/'manifest.json'
    if digest(manifest_path)!=expected_manifest_sha256: raise ValueError('dataset manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    protocol_path=evaluation/'protocol.json'; protocol=json.loads(protocol_path.read_text())
    if not (evaluation/'COMPLETE').is_file(): raise ValueError('evaluation is incomplete')
    if manifest.get('train_per_cell') != 200 or set(protocol['cells']) != set(CELLS):
        raise ValueError('unexpected dataset or cell protocol')
    # Ensure all dataset image hashes match the frozen manifest before use.
    loaded={}
    for row in manifest['records']:
        name=row['image']
        if Path(name).name != name or name in loaded: raise ValueError('invalid or duplicate dataset image')
        loaded[name]=load_rgb(dataset/name,expected_pixel=row['pixel_sha256'])
    out=[]
    for cell in CELLS:
        complexity, balance=cell.split('-')
        report_path=evaluation/cell/'report.json'; report=json.loads(report_path.read_text())
        if report['checkpoint_sha256'] != protocol['cells'][cell]['sha256']:
            raise ValueError('checkpoint mismatch between protocol and report')
        expected={(color,seed,1.0) for color in ('red','blue') for seed in protocol['seeds']}
        actual=[(item['color'],item['seed'],item['guidance']) for item in report['images']]
        if len(expected)!=16 or len(actual)!=16 or set(actual)!=expected:
            raise ValueError('expected exactly the 16 prespecified color/seed samples')
        for color in ('red','blue'):
            refs=[r for r in manifest['records'] if r['split']=='train' and r['complexity']==complexity
                  and r['imbalance']==balance and r['target_color']==color]
            controls=[r for r in manifest['records'] if r['split']=='evaluation' and r['complexity']==complexity
                      and r['target_color']==color]
            generated=[r for r in report['images'] if r['color']==color and r['guidance']==1.0]
            if not refs or not controls or len(generated)!=8: raise ValueError('missing reference, control or sample group')
            reference=np.stack([loaded[r['image']] for r in refs])
            results={}
            for label, items in [('generated', generated),('heldout_renderer_controls', controls)]:
                rows=[]
                for item in items:
                    filename=item['file'] if label=='generated' else item['image']
                    if Path(filename).name != filename: raise ValueError('invalid image path')
                    if label=='generated':
                        image_path=evaluation/cell/filename
                        if digest(image_path)!=item['sha256']: raise ValueError('generated file hash mismatch')
                        image=load_rgb(image_path,expected_pixel=item['pixel_sha256'])
                    else: image=loaded[filename]
                    index,mse=nearest_mse(image,reference)
                    rows.append({'image':filename,'nearest_training_image':refs[index]['image'],'mse':mse})
                results[label]={'summary':_summary(rows),'items':rows}
            out.append({'cell':cell,'target_color':color,'training_reference_count':len(refs),**results})
    return {'schema_version':1,'dataset_manifest_sha256':expected_manifest_sha256,
            'evaluation_protocol_sha256':digest(protocol_path),'metric':'mean squared error of 64x64 RGB pixels in [0,1]',
            'groups':out,'note':('Descriptive nearest neighbors only: no memorization classification or statistical inference. '
            'Renderer controls differ in appearance from model outputs; target colors and complexity are matched. '
            'Control images are reused across balance conditions; generated noise seeds are paired.')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:
        result=audit(a.dataset,a.evaluation,expected_manifest_sha256=a.expected_manifest_sha256)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    except (ValueError,KeyError,OSError) as exc: p.error(str(exc))
    print(json.dumps([{'cell':g['cell'],'color':g['target_color'],'generated':g['generated']['summary'],
                       'controls':g['heldout_renderer_controls']['summary']} for g in result['groups']],indent=2))

if __name__ == '__main__': main()
