"""Coverage (schema §9.2): R2_attr / R2_full per domain."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from src.probe import load_feat, align
from src.data import aadb_attr_matrix, ATTRS, baid_scores, ava_scores

READABLE = ['ColorHarmony','Content','DoF','Light','Object','Repetition','VividColor']
BACKBONES = ['dinov2','clip','siglip2']

def _xy(bb, domain, split):
    X, names = load_feat(bb, domain, split)
    sc = baid_scores(split) if domain == "baid" else ava_scores()
    keep = [i for i,n in enumerate(names) if n in sc]
    return X[keep], np.array([sc[names[i]] for i in keep])

def attr_probes(bb):
    Xtr, ftr = load_feat(bb, "aadb", "train")
    Mtr, ltr = aadb_attr_matrix("train")
    Xtr, Mtr = align(Xtr, ftr, ltr, Mtr)
    return {a: Ridge(alpha=10.0).fit(Xtr, Mtr[:, ATTRS.index(a)+1]) for a in READABLE}

def coverage(bb, domain):
    probes = attr_probes(bb)
    Xtr, ytr = _xy(bb, domain, "train")
    Xte, yte = _xy(bb, domain, "test")
    full = Ridge(alpha=10.0).fit(Xtr, ytr)
    r2_full = r2_score(yte, full.predict(Xte))
    Ptr = np.column_stack([probes[a].predict(Xtr) for a in READABLE])
    Pte = np.column_stack([probes[a].predict(Xte) for a in READABLE])
    attr = Ridge(alpha=10.0).fit(Ptr, ytr)
    r2_attr = r2_score(yte, attr.predict(Pte))
    return r2_full, r2_attr

if __name__ == "__main__":
    print(f"{'domain':6s} {'backbone':8s} {'R2_full':>8s} {'R2_attr':>8s} {'coverage':>9s}")
    agg = {}
    for dom in ["ava", "baid"]:
        cov = []
        for bb in BACKBONES:
            rf, ra = coverage(bb, dom)
            c = ra / rf if rf > 0 else float('nan')
            cov.append(c)
            print(f"{dom:6s} {bb:8s} {rf:8.3f} {ra:8.3f} {c:9.3f}")
        agg[dom] = np.nanmean(cov)
    print(f"\ncoverage(photo)={agg['ava']:.3f}  coverage(art)={agg['baid']:.3f}")
    print(f"headline gap = {agg['ava'] - agg['baid']:.3f}")
