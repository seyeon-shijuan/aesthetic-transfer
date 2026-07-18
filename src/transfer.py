"""통과 probe를 P/R 도메인에 적용 → r_a, T_a, shift_within."""
import numpy as np, scipy.io, scipy.stats
from src.probe import load_feat, align, BACKBONES
from src.data import aadb_attr_matrix, ATTRS, baid_scores, ava_scores

READABLE = ['ColorHarmony','Content','DoF','Light','Object','Repetition','VividColor']
CACHE = "/data2/seyeon/myjob/aesthetic-transfer/cache"

def r_of(bb, attr_i, domain):
    # AADB train으로 probe 학습
    Xtr, ftr = load_feat(bb,"aadb","train"); Mtr, ltr = aadb_attr_matrix("train")
    Xtr, Mtr = align(Xtr, ftr, ltr, Mtr)
    from sklearn.linear_model import Ridge
    g = Ridge(alpha=10.0).fit(Xtr, Mtr[:, attr_i+1])
    # 도메인 test feature에 적용
    X, names = load_feat(bb, domain, "test")
    pred = g.predict(X)
    # 그 도메인의 총점과 SRCC
    if domain == "baid":
        sc = baid_scores("test"); y = np.array([sc[n] for n in names])
    else:  # ava
        sc = ava_scores();        y = np.array([sc[n] for n in names])
    return scipy.stats.spearmanr(pred, y).correlation

if __name__ == "__main__":
    print(f"{'attr':14s} {'domain':6s} " + " ".join(f"{b:>7s}" for b in BACKBONES))
    for attr in READABLE:
        ai = ATTRS.index(attr)
        for dom in ["ava","baid"]:
            rs = [r_of(bb, ai, dom) for bb in BACKBONES]
            print(f"{attr:14s} {dom:6s} " + " ".join(f"{r:7.3f}" for r in rs))