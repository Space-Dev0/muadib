"""Generate descriptive figures from the completed 64-image Stage 0 review."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

data = json.loads((Path(__file__).parent / 'review-aggregate.json').read_text())
out = Path(__file__).parent / 'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10, 'savefig.bbox': 'tight', 'figure.dpi': 160})
labels = ['Low / balanced', 'Low / imbalanced', 'High / balanced', 'High / imbalanced']
fig, ax = plt.subplots(figsize=(7.4, 3.5))
y = np.arange(4)
left = np.zeros(4)
cells = ['low-balanced', 'low-imbalanced', 'high-balanced', 'high-imbalanced']
for name, key, color in [('Yes', 'yes', '#2a8a71'), ('No', 'no', '#bd5a4d'), ('Unresolved', 'unresolved', '#c5a34e')]:
    vals = [sum(data['cells_by_color'][cell][c][key] for c in ('red', 'blue')) for cell in cells]
    ax.barh(y, vals, left=left, label=name, color=color)
    for row, value, start in zip(y, vals, left):
        if value:
            ax.text(start + value / 2, row, str(value), ha='center', va='center',
                    fontsize=8, color='white' if name != 'Unresolved' else 'black')
    left += vals
ax.set(yticks=y, yticklabels=labels, xlabel='Reviewed images (16 per cell)', xlim=(0,16))
ax.invert_yaxis(); ax.legend(ncol=3, loc='lower right'); ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'review-counts.pdf'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.8, 2.4))
ax.barh(['Standard checker','Experimental checker'], [data['checker_comparison'][c]['agree'] for c in ('standard', 'variable_count')], color='#4a75a8', label='Answered')
ax.barh(['Standard checker','Experimental checker'], [data['checker_comparison'][c]['abstained'] for c in ('standard', 'variable_count')], left=[data['checker_comparison'][c]['agree'] for c in ('standard', 'variable_count')], color='#e3b664', label='Abstained')
ax.set(xlim=(0,64), xlabel='Images (out of 64)'); ax.legend(ncol=2, loc='lower right'); ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'checker-coverage.pdf'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.8, 2.8))
for i,(name,lo,hi) in enumerate([('Complexity drop', *data['complexity_drop_bounds']), ('Imbalance drop', *data['imbalance_drop_bounds']), ('Difference', *data['difference_in_drops_bounds'])]):
    ax.plot([lo,hi],[i,i],linewidth=7,color='#476c9b',solid_capstyle='butt'); ax.scatter([lo,hi],[i,i],color='#244160',s=12)
ax.axvline(0,color='black',lw=.7); ax.set(yticks=range(3),yticklabels=['Complexity drop','Imbalance drop','Difference'],xlabel='Finite-image bounds (not confidence intervals)'); ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'bounds.pdf'); plt.close(fig)
base=Path(__file__).parent / 'sample-images'
fig, axes=plt.subplots(1,4,figsize=(7.4,2.1))
for ax,cell,title in zip(axes,['low-balanced','low-imbalanced','high-balanced','high-imbalanced'],labels):
    ax.imshow(Image.open(base/(cell + '.png')).convert('RGB')); ax.set_title(title,fontsize=8); ax.axis('off')
fig.savefig(out/'matched-samples.pdf'); plt.close(fig)

# Explanatory example: two binary distributions have identical marginals but
# different joint probability. These are constructed examples, not data.
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)
for ax, vals, title in zip(axes, [np.array([[.1, .4], [.4, .1]]), np.array([[.4, .1], [.1, .4]])], ['Low co-occurrence', 'High co-occurrence']):
    ax.imshow(vals, cmap='Blues', vmin=0, vmax=.5)
    ax.set(xticks=[0, 1], xticklabels=['B absent', 'B present'], yticks=[0, 1], yticklabels=['A absent', 'A present'], title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{vals[i,j]:.1f}', ha='center', va='center', color='white' if vals[i,j] > .3 else 'black')
    ax.text(.5, -0.16, f'P(A)=P(B)=0.5; P(A,B)={vals[1,1]:.1f}', ha='center', transform=ax.transAxes, fontsize=9)
fig.savefig(out/'marginals-joint-example.pdf'); plt.close(fig)

# The presentation's fixed Gaussian-noise blend is illustrative only: it is
# not an actual forward sample from a specified DDPM schedule.
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9), constrained_layout=True)
for ax, fraction, idx in zip(axes, ['0%', '35%', '70%', '100%'], range(4)):
    ax.imshow(Image.open(base / f'noise-{idx}.png').convert('RGB'))
    ax.set_title(f'Noise blend: {fraction}', fontsize=9)
    ax.axis('off')
fig.savefig(out/'illustrative-noising.pdf'); plt.close(fig)

# Fixed-seed generated images selected for known review outcomes, from the
# frozen blinded pack; they are examples, not a representative sample.
fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.5), constrained_layout=True)
for ax, item, title in zip(axes, ['0001', '0008', '0043'], ['Blue request: unsure', 'Blue request: no', 'Red request: no']):
    ax.imshow(Image.open(base / f'matched-{item}.png').convert('RGB'))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Review item ' + item, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(out/'review-failures.pdf'); plt.close(fig)

# Color-stratified counts from score-v2.json, with unresolved retained.
fig, ax = plt.subplots(figsize=(7.3, 4.0), constrained_layout=True)
colors = ['red', 'blue'] * 4
y = np.arange(8)
rows = [data['cells_by_color'][cell][color] for cell in cells for color in ('red', 'blue')]
yes, no, unsure = (np.array([row[key] for row in rows]) for key in ('yes', 'no', 'unresolved'))
ax.barh(y, yes, color='#2a8a71', label='Yes')
ax.barh(y, no, left=yes, color='#bd5a4d', label='No')
ax.barh(y, unsure, left=yes + no, color='#c5a34e', label='Unresolved')
ax.set(yticks=y, yticklabels=[f'{cell} / {color}' for cell in labels for color in ('red','blue')], xlim=(0, 8), xlabel='Images per condition and request (8 total)')
ax.invert_yaxis(); ax.legend(ncol=3, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
fig.savefig(out/'review-by-color.pdf'); plt.close(fig)
