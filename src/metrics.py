"""Kendall's W across backbones on T_a ranking."""
import numpy as np
from scipy.stats import rankdata

# transfer.txt에서 백본별 T 재구성
rows = {}
for l in open("results/transfer.txt"):
    p = l.split()
    if len(p)==5 and p[1] in ("ava","baid"):
        rows.setdefault(p[0],{})[p[1]] = np.array([float(x) for x in p[2:]])

attrs = list(rows)
# T per backbone: (n_attr, 3)
T = np.array([rows[a]['ava'] - rows[a]['baid'] for a in attrs])  # (7,3)

def kendall_w(T):
    # 각 백본(열)이 속성(행)에 매긴 순위
    R = np.apply_along_axis(rankdata, 0, T)   # rank within each backbone
    n, m = R.shape                            # n attrs, m backbones
    Rsum = R.sum(1)
    S = ((Rsum - Rsum.mean())**2).sum()
    return 12*S / (m**2 * (n**3 - n))

W = kendall_w(T)
print(f"attrs: {attrs}")
print(f"T per backbone (rows=attr, cols=dino/clip/siglip):\n{np.round(T,3)}")
print(f"\nKendall's W = {W:.3f}")
print("(1=완전일치, 0=무작위. >0.7이면 백본 무관하게 순위 안정)")