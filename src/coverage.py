"""Coverage: how much aesthetic variance the readable attributes explain, per domain.
사용: python -m src.coverage"""
import numpy as np, scipy.stats
from sklearn.linear_model import Ridge, LinearRegression
from src.probe import load_feat, align
from src.data import aadb_attr_matrix, ATTRS, baid_scores, ava_scores

READABLE = ['ColorHarmony','Content','DoF','Light','Object','Repetition','VividColor']
BACKBONES = ['dinov2','clip','siglip2']

def probe_predictions(bb, domain):
    """7개 readable 속성 probe를 학습해 domain test에 적용 → (n, 7) 예측행렬 + 총점 y."""
    Xtr, ftr = load_feat(bb, "aadb", "train")
    Mtr, ltr = aadb_attr_matrix("train")
    Xtr, Mtr = align(Xtr, ftr, ltr, Mtr)

    X, names = load_feat(bb, domain, "test")
    sc = baid_scores("test") if domain == "baid" else ava_scores()
    keep = [i for i,n in enumerate(names) if n in sc]
    X, y = X[keep], np.array([sc[names[i]] for i in keep])

    preds = []
    for attr in READABLE:
        ai = ATTRS.index(attr)
        g = Ridge(alpha=10.0).fit(Xtr, Mtr[:, ai+1])
        preds.append(g.predict(X))
    return np.column_stack(preds), y   # (n,7), (n,)

def r2_via_rank(P, y):
    """예측 속성들로 총점을 설명하는 정도. rank 기반이라 스케일 무관.
       Spearman과 정합되게, rank(y)를 rank(P)들의 선형결합으로 회귀 → R²."""
    Pr = np.column_stack([scipy.stats.rankdata(P[:,j]) for j in range(P.shape[1])])
    yr = scipy.stats.rankdata(y)
    lr = LinearRegression().fit(Pr, yr)
    return lr.score(Pr, yr)   # R² on ranks

if __name__ == "__main__":
    print(f"{'domain':6s} " + " ".join(f"{b:>8s}" for b in BACKBONES) + f"   {'mean':>8s}")
    for dom in ["ava", "baid"]:
        r2s = []
        for bb in BACKBONES:
            P, y = probe_predictions(bb, dom)
            r2s.append(r2_via_rank(P, y))
        print(f"{dom:6s} " + " ".join(f"{v:8.3f}" for v in r2s) + f"   {np.mean(r2s):8.3f}")
    print("\nR^2(rank): 7 readable attributes 로 총점 rank를 설명하는 비율. photo↓art 이면 어휘 전이 실패.")