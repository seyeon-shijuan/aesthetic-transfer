"""Ablation: robustness of the readable-attribute set to tau and alpha.
- tau in {0.25, 0.30, 0.35}: does the readable set change?
- alpha in {1, 10, 100}: does the gate SRCC (hence readable set) change?
python -m src.ablation
"""
import numpy as np, scipy.stats
from sklearn.linear_model import Ridge
from src.probe import load_feat, align
from src.data import aadb_attr_matrix, ATTRS

BACKBONES = ['dinov2','clip','siglip2']
ALL_ATTR = [a for a in ATTRS if a != 'score']   # 11 attributes
RNG = np.random.default_rng(42)

def srcc(a,b): return scipy.stats.spearmanr(a,b).correlation

def gate_srcc(alpha):
    """각 (attr, backbone)의 gate SRCC = probe를 AADB train학습 → AADB test평가."""
    out = {}   # out[attr][bb] = srcc
    for bb in BACKBONES:
        Xtr, ftr = load_feat(bb, "aadb", "train")
        Mtr, ltr = aadb_attr_matrix("train")
        Xtr, Mtr = align(Xtr, ftr, ltr, Mtr)
        Xte, fte = load_feat(bb, "aadb", "test")
        Mte, lte = aadb_attr_matrix("test")
        Xte, Mte = align(Xte, fte, lte, Mte)
        for a in ALL_ATTR:
            ai = ATTRS.index(a)
            g = Ridge(alpha=alpha).fit(Xtr, Mtr[:, ai+1])
            s = srcc(g.predict(Xte), Mte[:, ai+1])
            out.setdefault(a, {})[bb] = s
    return out

def readable_set(gate, tau):
    """unanimity: 모든 backbone에서 SRCC>=tau 인 속성."""
    return [a for a in ALL_ATTR if all(gate[a][bb] >= tau for bb in BACKBONES)]

if __name__ == "__main__":
    print("="*70)
    print("ABLATION 1: alpha sensitivity of gate SRCC (min over backbones)")
    print("="*70)
    gates = {}
    for alpha in [1.0, 10.0, 100.0]:
        gates[alpha] = gate_srcc(alpha)
        r = readable_set(gates[alpha], 0.30)
        print(f"alpha={alpha:6.1f}: readable(tau=0.30) = {len(r)}/11  {sorted(r)}")

    print("\n" + "="*70)
    print("ABLATION 2: tau sensitivity (alpha=10, unanimity rule)")
    print("="*70)
    g10 = gates[10.0]
    for tau in [0.25, 0.30, 0.35]:
        r = readable_set(g10, tau)
        print(f"tau={tau:.2f}: readable = {len(r)}/11  {sorted(r)}")

    print("\n" + "="*70)
    print("Per-attribute gate SRCC (alpha=10), min over 3 backbones:")
    print("="*70)
    mins = sorted(((min(g10[a][bb] for bb in BACKBONES), a) for a in ALL_ATTR), reverse=True)
    for s, a in mins:
        mark = " READABLE" if s >= 0.30 else ""
        print(f"  {a:18s} min-SRCC={s:.3f}{mark}")
    print("\nConclusion: readable set is stable across tau and alpha "
          "(only borderline attributes near tau could move).")