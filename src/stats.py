"""Statistical robustness — FAST version.
Key fix: fit each probe ONCE, cache predictions, bootstrap resamples indices only.
사용: python -m src.stats
"""
import numpy as np, scipy.stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from src.probe import load_feat, align
from src.data import aadb_attr_matrix, ATTRS, baid_scores, ava_scores

CACHE = "/data2/seyeon/myjob/aesthetic-transfer/cache"
READABLE = ['ColorHarmony','Content','DoF','Light','Object','Repetition','VividColor']
BACKBONES = ['dinov2','clip','siglip2']
RNG = np.random.default_rng(42)
NBOOT = 2000

def srcc(a, b):
    return scipy.stats.spearmanr(a, b).correlation

# ---------------------------------------------------------------- precompute ALL probe predictions once
print("Precomputing probes (once)...", flush=True)
# probes[attr][bb] = trained ridge ; predictions cached per (attr,bb,domain)
PRED = {}   # PRED[(attr,bb,domain)] = (pred, gt)
FULL = {}   # FULL[(bb,domain)] = (X_test, y_test, X_train, y_train) for coverage
_probe = {}
for bb in BACKBONES:
    Xtr, ftr = load_feat(bb, "aadb", "train")
    Mtr, ltr = aadb_attr_matrix("train")
    Xtr, Mtr = align(Xtr, ftr, ltr, Mtr)
    _probe[bb] = {a: Ridge(alpha=10.0).fit(Xtr, Mtr[:, ATTRS.index(a)+1]) for a in READABLE}
    for domain in ["ava","baid"]:
        X, names = load_feat(bb, domain, "test")
        sc = baid_scores("test") if domain=="baid" else ava_scores()
        keep = [i for i,n in enumerate(names) if n in sc]
        Xk = X[keep]; yk = np.array([sc[names[i]] for i in keep])
        for a in READABLE:
            PRED[(a,bb,domain)] = (_probe[bb][a].predict(Xk), yk)
print("done.\n", flush=True)

# ---------------------------------------------------------------- 1. Transfer shift CI + reversal test
def transfer_ci():
    print("="*72)
    print("1. TRANSFER SHIFT  T_a = r_photo - r_art   (bootstrap 95% CI over 3 backbones)")
    print("="*72)
    print(f"{'attr':14s} {'T_a':>7s} {'95% CI':>18s} {'reversal p':>11s}")
    out = {}
    n_ava = len(PRED[(READABLE[0],'dinov2','ava')][1])
    n_baid = len(PRED[(READABLE[0],'dinov2','baid')][1])
    for attr in READABLE:
        vals = np.empty(NBOOT)
        for k in range(NBOOT):
            ia = RNG.integers(0, n_ava, n_ava)
            ib = RNG.integers(0, n_baid, n_baid)
            r_ava = np.mean([srcc(PRED[(attr,b,'ava')][0][ia], PRED[(attr,b,'ava')][1][ia]) for b in BACKBONES])
            r_baid = np.mean([srcc(PRED[(attr,b,'baid')][0][ib], PRED[(attr,b,'baid')][1][ib]) for b in BACKBONES])
            vals[k] = r_ava - r_baid
        mean = vals.mean(); lo, hi = np.percentile(vals, [2.5, 97.5])
        p_rev = np.mean(vals >= 0) if mean < 0 else np.mean(vals <= 0)
        out[attr] = (mean, lo, hi, p_rev)
        tag = "  <-- reversal" if mean < 0 else ""
        print(f"{attr:14s} {mean:+7.3f} [{lo:+.3f}, {hi:+.3f}] {p_rev:11.4f}{tag}", flush=True)
    return out

# ---------------------------------------------------------------- 2. Kendall's W permutation
def kendall_w_perm():
    print("\n" + "="*72)
    print("2. KENDALL'S W  permutation test")
    print("="*72)
    rows = {}
    for l in open("results/transfer.txt"):
        p = l.split()
        if len(p)==5 and p[1] in ("ava","baid"):
            rows.setdefault(p[0],{})[p[1]] = np.array([float(x) for x in p[2:]])
    attrs = list(rows)
    T = np.array([rows[a]['ava'] - rows[a]['baid'] for a in attrs])
    from scipy.stats import rankdata
    def kw(mat):
        R = np.apply_along_axis(rankdata, 0, mat); n,m = R.shape
        Rsum = R.sum(1); S = ((Rsum-Rsum.mean())**2).sum()
        return 12*S/(m**2*(n**3-n))
    W_obs = kw(T)
    perm = np.array([kw(np.array([RNG.permutation(T[:,j]) for j in range(T.shape[1])]).T) for _ in range(5000)])
    p = np.mean(perm >= W_obs)
    print(f"W_observed = {W_obs:.4f}")
    print(f"null mean = {perm.mean():.4f} | p-value = {p:.5f} (n=5000)")
    print(f"  -> {'highly significant' if p<0.001 else 'significant' if p<0.05 else 'not significant'}", flush=True)
    return W_obs, p

# ---------------------------------------------------------------- 3. Grid SRCC CI + pairwise
def grid_ci():
    print("\n" + "="*72)
    print("3. MODEL GRID  art (BAID) SRCC  bootstrap 95% CI")
    print("="*72)
    names = [n for n in open(f"{CACHE}/dinov2_baid_test_names.txt").read().split("\n") if n]
    sc = baid_scores("test")
    names = [n for n in names if n in sc]
    gt = np.array([sc[n] for n in names])
    label = {"qalign":"Q-Align (photo)","aes":"Aes-V2.5 (photo)","artimuse":"ArtiMuse (multi)"}
    store = {}
    for name in ["qalign","aes","artimuse"]:
        try: preds = np.load(f"{CACHE}/{name}_baid_scores.npy")
        except FileNotFoundError:
            print(f"  (skip {name})"); continue
        # names 파일이 있으면 그걸로 정확히 정렬 (파싱 스킵 대응)
        import os
        nf = f"{CACHE}/{name}_baid_names.txt"
        if os.path.exists(nf):
            kn = [x for x in open(nf).read().split("\n") if x]
            g = np.array([sc[x] for x in kn])
            p = preds
            n = min(len(p), len(g)); p, g = p[:n], g[:n]
        else:
            n = min(len(preds), len(gt)); p, g = preds[:n], gt[:n]
        vals = np.array([srcc(p[idx], g[idx]) for idx in (RNG.integers(0,n,n) for _ in range(NBOOT))])
        store[name] = vals
        print(f"{label[name]:20s} SRCC={srcc(p,g):.3f}  95% CI [{np.percentile(vals,2.5):.3f}, {np.percentile(vals,97.5):.3f}]", flush=True)
    if "artimuse" in store and "qalign" in store:
        m = min(len(store["artimuse"]), len(store["qalign"]))
        diff = store["artimuse"][:m] - store["qalign"][:m]
        p = np.mean(diff <= 0)
        print(f"\n  ArtiMuse vs Q-Align: mean diff={diff.mean():+.3f}, P(Arti<=QAlign)={p:.4f} "
              f"{'-> ArtiMuse significantly higher' if p<0.05 else '-> n.s.'}", flush=True)

# ---------------------------------------------------------------- 4. Coverage CI (fast: cache probe preds)
def coverage_ci():
    print("\n" + "="*72)
    print("4. COVERAGE  R2_attr/R2_full  bootstrap 95% CI")
    print("="*72)
    # precompute per-domain: full-feature train/test + attr-pred train/test
    for domain in ["ava","baid"]:
        rf_list, ra_list = [], []
        # need train too; build once per backbone
        packs = []
        for bb in BACKBONES:
            Xtr, ftr = load_feat(bb, "aadb", "train")
            Mtr, ltr = aadb_attr_matrix("train"); Xtr, Mtr = align(Xtr, ftr, ltr, Mtr)
            probes = {a: Ridge(alpha=10.0).fit(Xtr, Mtr[:, ATTRS.index(a)+1]) for a in READABLE}
            def xy(split):
                X, names = load_feat(bb, domain, split)
                sc = baid_scores(split) if domain=="baid" else ava_scores()
                keep = [i for i,n in enumerate(names) if n in sc]
                return X[keep], np.array([sc[names[i]] for i in keep])
            Xtr2, ytr = xy("train"); Xte, yte = xy("test")
            full = Ridge(alpha=10.0).fit(Xtr2, ytr)
            full_pred_te = full.predict(Xte)
            Ptr = np.column_stack([probes[a].predict(Xtr2) for a in READABLE])
            Pte = np.column_stack([probes[a].predict(Xte) for a in READABLE])
            attr = Ridge(alpha=10.0).fit(Ptr, ytr)
            attr_pred_te = attr.predict(Pte)
            packs.append((yte, full_pred_te, attr_pred_te))
        n = len(packs[0][0])
        point = np.nanmean([r2_score(y,fp)>0 and r2_score(y,ap)/r2_score(y,fp) or np.nan
                            for (y,fp,ap) in packs])
        vals = []
        for _ in range(1000):
            idx = RNG.integers(0, n, n)
            cs = []
            for (y,fp,ap) in packs:
                rf = r2_score(y[idx], fp[idx]); ra = r2_score(y[idx], ap[idx])
                cs.append(ra/rf if rf>0 else np.nan)
            vals.append(np.nanmean(cs))
        vals = np.array(vals); lo,hi = np.nanpercentile(vals,[2.5,97.5])
        print(f"{domain:6s} coverage={point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]", flush=True)

if __name__ == "__main__":
    transfer_ci()
    kendall_w_perm()
    grid_ci()
    coverage_ci()
    print("\nDone.", flush=True)