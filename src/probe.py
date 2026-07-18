"""Attribute probes + readability gate.
사용: python -m src.probe"""
import numpy as np, scipy.io, scipy.stats, glob, os
from sklearn.linear_model import Ridge

CACHE = "/data2/seyeon/myjob/aesthetic-transfer/cache"
DS = "/data2/seyeon/datasets"
ATTRS = ["BalacingElements","ColorHarmony","Content","DoF","Light",
         "MotionBlur","Object","Repetition","RuleOfThirds","Symmetry","VividColor"]
BACKBONES = ["dinov2","clip","siglip2"]
TAU = 0.30

def load_feat(bb, ds, split):
    X = np.load(f"{CACHE}/{bb}_{ds}_{split}.npy")
    names = open(f"{CACHE}/{bb}_{ds}_{split}_names.txt").read().split("\n")
    return X, names

def aadb_attr_matrix(split):
    """attMat 컬럼 + 그 순서에 대응하는 이미지 이름."""
    cell = {"train":0,"test":1}[split]
    M = scipy.io.loadmat(f"{DS}/AADB/labels/imgListFiles_label/attMat.mat")["dataset"][0][cell]
    # attMat 행 순서 = score 리스트 파일 순서
    S = {"train":"Train","test":"TestNew"}[split]
    f = f"{DS}/AADB/labels/imgListFiles_label/imgList{S}Regression_score.txt"
    names = [l.split()[0] for l in open(f) if l.strip()]
    return M, names   # M[:,0]=score, M[:,1:]=11 attrs; names[i] ↔ M[i]

def align(X, feat_names, label_names, M):
    """feature를 label 순서에 맞춰 재정렬."""
    pos = {n:i for i,n in enumerate(feat_names)}
    idx = [pos[n] for n in label_names if n in pos]
    keep = [n in pos for n in label_names]
    return X[idx], M[np.array(keep)]

def main():
    print(f"{'attr':16s} " + " ".join(f"{b:>8s}" for b in BACKBONES) + "   verdict")
    results = {}
    for ai, attr in enumerate(ATTRS):
        gates = {}
        for bb in BACKBONES:
            Xtr, ftr = load_feat(bb,"aadb","train")
            Mtr, ltr = aadb_attr_matrix("train")
            Xtr, Mtr = align(Xtr, ftr, ltr, Mtr)

            Xte, fte = load_feat(bb,"aadb","test")
            Mte, lte = aadb_attr_matrix("test")
            Xte, Mte = align(Xte, fte, lte, Mte)

            # g = Ridge(alpha=1.0).fit(Xtr, Mtr[:, ai+1])
            g = Ridge(alpha=10.0).fit(Xtr, Mtr[:, ai+1])
            
            pred = g.predict(Xte)
            gates[bb] = scipy.stats.spearmanr(pred, Mte[:, ai+1]).correlation
        ok = all(v >= TAU for v in gates.values())
        results[attr] = (gates, ok)
        print(f"{attr:16s} " + " ".join(f"{gates[b]:8.3f}" for b in BACKBONES)
              + f"   {'READABLE' if ok else 'unreadable'}")
    readable = [a for a,(g,ok) in results.items() if ok]
    print(f"\nτ={TAU} · 3백본 만장일치 · READABLE {len(readable)}/11: {readable}")

if __name__ == "__main__":
    main()