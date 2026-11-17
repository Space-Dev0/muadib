"""Generate descriptive figures from the completed 64-image Stage 0 review."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

out = Path(__file__).parent / 'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'savefig.bbox': 'tight', 'figure.dpi': 160})
labels = ['Low / balanced', 'Low / imbalanced', 'High / balanced', 'High / imbalanced']
fig, ax = plt.subplots(figsize=(7.4, 3.5))
y = np.arange(4)
left = np.zeros(4)
for name, vals, color in [('Yes', [16,15,14,14], '#2a8a71'), ('No', [0,0,2,1], '#bd5a4d'), ('Unresolved', [0,1,0,1], '#c5a34e')]:
    ax.barh(y, vals, left=left, label=name, color=color)
    left += vals
ax.set(yticks=y, yticklabels=labels, xlabel='Reviewed images (16 per cell)', xlim=(0,16))
ax.invert_yaxis(); ax.legend(ncol=3, loc='lower right'); ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'review-counts.pdf'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.8, 2.4))
ax.barh(['Standard checker','Experimental checker'], [9,46], color='#4a75a8', label='Answered')
ax.barh(['Standard checker','Experimental checker'], [55,18], left=[9,46], color='#e3b664', label='Abstained')
ax.set(xlim=(0,64), xlabel='Images (out of 64)'); ax.legend(ncol=2, loc='lower right'); ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'checker-coverage.pdf'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.8, 2.8))
for i,(name,lo,hi) in enumerate([('Complexity drop',.0625,.125),('Imbalance drop',-.03125,.03125),('Difference',.0625,.125)]):
    ax.plot([lo,hi],[i,i],linewidth=7,color='#476c9b',solid_capstyle='butt'); ax.scatter([lo,hi],[i,i],color='#244160',s=12)
ax.axvline(0,color='black',lw=.7); ax.set(yticks=range(3),yticklabels=['Complexity drop','Imbalance drop','Difference'],xlabel='Finite-image bounds (not confidence intervals)'); ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'bounds.pdf'); plt.close(fig)
base=Path(__file__).parent / 'sample-images'
fig, axes=plt.subplots(1,4,figsize=(7.4,2.1))
for ax,cell,title in zip(axes,['low-balanced','low-imbalanced','high-balanced','high-imbalanced'],labels):
    ax.imshow(Image.open(base/(cell + '.png')).convert('RGB')); ax.set_title(title,fontsize=8); ax.axis('off')
fig.savefig(out/'matched-samples.pdf'); plt.close(fig)
