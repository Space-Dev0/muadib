"""Rebuild the course presentation's small, explicitly scoped figures.

Run from the repository root with matplotlib and Pillow installed. The FFT
values were computed by exposure.image_diagnostics from the two supplied images.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
assets = ROOT / 'assets'
network = json.loads((ROOT.parent / 'ops/extended-local-benchmark.json').read_text())
fig, ax = plt.subplots(figsize=(6, 2.5))
rows = network['measurements']
tunnel = [json.loads(line) for line in (ROOT.parent / 'ops/tunnel-benchmark.jsonl').read_text().splitlines()]
x = np.arange(len(rows))
ax.bar(x - .18, [r['scenes_per_second'] for r in rows], width=.36, color='#316a8a', label='Localhost')
ax.bar(x + .18, [r['scenes_per_second'] for r in tunnel], width=.36, color='#c38645', label='SSH tunnel to laptop')
ax.set(xlabel='Scenes per batch', ylabel='Scenes / second', ylim=(0, 1400), xticks=x, xticklabels=[str(r['batch_size']) for r in rows])
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(assets / 'localhost-stream.png', dpi=150)
plt.close(fig)

spectra = json.loads((assets / 'spectrum-seed7.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), gridspec_kw={'width_ratios': [1, 1, 2]})
for ax, name, title in zip(axes[:2], ['spectral-clean-seed7.png', 'spectral-blur-seed7.png'],
                           ['Clean render', 'Blur radius 1']):
    ax.imshow(Image.open(assets / name)); ax.set_title(title); ax.axis('off')
for key, color in [('clean', '#326b95'), ('blur', '#bb7542')]:
    values = spectra[key + '_mean_power']
    axes[2].semilogy(spectra['frequency_radius_bins'][1:], [max(v, 1e-12) for v in values[1:]],
                     label=key, color=color)
axes[2].set(xlabel='Frequency radius (bins)', ylabel='Mean radial power')
axes[2].legend(); axes[2].spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(assets / 'paired-spectral-demo.png', dpi=160)
plt.close(fig)
