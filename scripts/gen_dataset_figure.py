# -*- coding: utf-8 -*-
"""Generate a WIDE dataset-sample montage for the paper (English labels).

3 examples (rows) x 6 classes (columns), each column titled with the class
name + sample count. Wide layout suits a full-width figure* float.
Saves submit/figures/dataset_samples.png.
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "magnetic_tile")
OUT = os.path.join(ROOT, "submit", "figures")
os.makedirs(OUT, exist_ok=True)

CLASS_DIRS = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Free", "MT_Uneven"]
CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]
N_EXAMPLES = 3


def pick(paths):
    n = len(paths)
    idx = [0, n // 2, n - 1] if n >= N_EXAMPLES else list(range(n))
    return [paths[i] for i in idx]


fig, axes = plt.subplots(N_EXAMPLES, len(CLASSES), figsize=(12, 5.6))
for c, (cdir, cname) in enumerate(zip(CLASS_DIRS, CLASSES)):
    paths = sorted(glob.glob(os.path.join(DATA, cdir, "*.jpg")))
    ex = pick(paths)
    for r in range(N_EXAMPLES):
        ax = axes[r, c]
        if r < len(ex):
            img = np.asarray(Image.open(ex[r]).convert("L"))
            ax.imshow(img, cmap="gray", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, c].set_title(f"{cname}\n({len(paths)})", fontsize=9)

fig.suptitle("Representative MTSD samples (three examples per class)",
             fontsize=11, y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "dataset_samples.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved dataset_samples.png (wide)")
