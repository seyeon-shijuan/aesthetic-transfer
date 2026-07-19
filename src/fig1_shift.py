"""Figure 1: per-attribute transfer shift (photo vs art)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# transfer.txt 파싱 → 3백본 평균 r_ava, r_baid
rows = {}
for l in open("results/transfer.txt"):
    p = l.split()
    if len(p) == 5 and p[1] in ("ava", "baid"):
        rows.setdefault(p[0], {})[p[1]] = np.mean([float(x) for x in p[2:]])

attrs = list(rows)
T = {a: rows[a]["ava"] - rows[a]["baid"] for a in attrs}
order = sorted(attrs, key=lambda a: T[a], reverse=True)   # shift 큰 순
vals = [T[a] for a in order]
colors = ["#d1495b" if v > 0 else "#3a7ca5" for v in vals]  # +빨강(photo-spec) / -파랑(art-strong)

fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.barh(range(len(order)), vals, color=colors)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel(r"Transfer shift  $T_a = r_a(\mathrm{photo}) - r_a(\mathrm{art})$")
ax.set_title("Photographic aesthetic attributes lose relevance in art")

for i, v in enumerate(vals):
    offset = 0.012
    ax.text(v + (offset if v >= 0 else -offset), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlim(-0.16, 0.40) 
# 범례
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#d1495b", label="Photo-specific (T>0)"),
                   Patch(color="#3a7ca5", label="Stronger in art (T<0)")],
          loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("results/fig1_shift.png", dpi=200, bbox_inches="tight")
plt.savefig("results/fig1_shift.pdf", bbox_inches="tight")
print("saved results/fig1_shift.png / .pdf")