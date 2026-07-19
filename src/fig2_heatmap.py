import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# transfer.txt에서 백본별 T 재구성
rows = {}
for l in open("results/transfer.txt"):
    p = l.split()
    if len(p)==5 and p[1] in ("ava","baid"):
        rows.setdefault(p[0],{})[p[1]] = np.array([float(x) for x in p[2:]])
attrs = list(rows)
backbones = ["DINOv2","CLIP","SigLIP2"]
T = np.array([rows[a]['ava'] - rows[a]['baid'] for a in attrs])  # (7,3)
order = np.argsort(-T.mean(1))
T, attrs = T[order], [attrs[i] for i in order]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
vmax = np.abs(T).max()
im = ax.imshow(T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(backbones)
ax.set_yticks(range(len(attrs))); ax.set_yticklabels(attrs)
for i in range(len(attrs)):
    for j in range(3):
        ax.text(j, i, f"{T[i,j]:+.2f}", ha="center", va="center",
                color="white" if abs(T[i,j])>vmax*0.5 else "black", fontsize=9)
ax.set_title("Transfer shift $T_a$ per backbone (Kendall's $W$=0.849)")
plt.colorbar(im, label="$T_a$ = r(photo) − r(art)")
plt.tight_layout()
plt.savefig("results/fig2_heatmap.png", dpi=200, bbox_inches="tight")
plt.savefig("results/fig2_heatmap.pdf", bbox_inches="tight")
print("saved fig2")