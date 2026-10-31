# -*- coding: utf-8 -*-
"""生成论文图：MTSD 长尾分布 + 漏检/误报代价不对称（Fig. 1）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

classes = ["Free", "Blowhole", "Uneven", "Break", "Crack", "Fray"]
counts = [952, 115, 103, 85, 57, 32]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

# (a) 长尾分布
ax = axes[0]
order = np.argsort(counts)[::-1]
cls = [classes[i] for i in order]
cnt = [counts[i] for i in order]
col = [colors[i] for i in order]
bars = ax.bar(cls, cnt, color=col, edgecolor="black", linewidth=0.6)
ax.set_ylabel("Number of samples", fontsize=9)
ax.set_title("(a) Long-tailed class distribution (MTSD)", fontsize=10)
for b, c in zip(bars, cnt):
    ax.text(b.get_x() + b.get_width() / 2, c + 15, str(c), ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 1050)
ax.tick_params(axis="x", labelsize=8, rotation=25)
ax.tick_params(axis="y", labelsize=8)
ax.annotate("imbalance ratio\n~ 30 : 1", xy=(0.0, 952), xytext=(0.85, 760),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8), ha="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (b) 代价不对称
ax = axes[1]
ax.axis("off")
ax.set_title("(b) Asymmetric error costs (quality-cost view)", fontsize=10)
ax.add_patch(plt.Rectangle((0.08, 0.30), 0.34, 0.18, facecolor="#D8E4F0",
                           edgecolor="black", lw=0.8))
ax.text(0.25, 0.46, "False alarm", ha="center", va="center", fontsize=8.5, weight="bold")
ax.text(0.25, 0.39, "normal -> defect", ha="center", va="center", fontsize=7.5)
ax.text(0.25, 0.33, "one extra re-inspection\n(appraisal cost, low)",
        ha="center", va="center", fontsize=7)
ax.add_patch(plt.Rectangle((0.58, 0.05), 0.34, 0.62, facecolor="#F5C6C6",
                           edgecolor="black", lw=0.8))
ax.text(0.75, 0.58, "Miss", ha="center", va="center", fontsize=8.5, weight="bold")
ax.text(0.75, 0.49, "defect -> normal", ha="center", va="center", fontsize=7.5)
ax.text(0.75, 0.36, "defective product flows\ndownstream / to market",
        ha="center", va="center", fontsize=7)
ax.text(0.75, 0.16, "external failure cost\n(the highest)",
        ha="center", va="center", fontsize=7, weight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.72)
ax.text(0.5, 0.70, "macro G-mean protects the weakest (highest-cost) class",
        ha="center", fontsize=8, style="italic")

fig.tight_layout()
fig.savefig("ESWA_tools-main/LaTeXSource/longtail_cost.png", dpi=200, bbox_inches="tight")
print("saved longtail_cost.png")
