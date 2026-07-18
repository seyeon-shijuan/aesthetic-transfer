"""Label loaders for AADB / BAID / AVA."""
import numpy as np, pandas as pd, scipy.io

DS = "/data2/seyeon/datasets"
ATTRS = ["BalacingElements","ColorHarmony","Content","DoF","Light",
         "MotionBlur","Object","Repetition","RuleOfThirds","Symmetry","VividColor"]

def aadb_attr_matrix(split):
    """returns (M, names): M[:,0]=score, M[:,1:12]=attrs; names[i] ↔ M[i]."""
    cell = {"train":0,"test":1}[split]
    M = scipy.io.loadmat(f"{DS}/AADB/labels/imgListFiles_label/attMat.mat")["dataset"][0][cell]
    S = {"train":"Train","test":"TestNew"}[split]
    f = f"{DS}/AADB/labels/imgListFiles_label/imgList{S}Regression_score.txt"
    names = [l.split()[0] for l in open(f) if l.strip()]
    return M, names

def baid_scores(split="test"):
    df = pd.read_csv(f"{DS}/BAID/dataset/{split}_set.csv")
    return dict(zip(df["image"], df["score"].astype(float)))

def ava_scores():
    """AVA.txt: idx id c1..c10 tags... → weighted-mean score keyed by '<id>.jpg'."""
    s = {}
    for l in open(f"{DS}/AVA_raw/AVA_dataset/AVA.txt"):
        p = l.split()
        c = np.array([int(x) for x in p[2:12]], float)
        s[f"{p[1]}.jpg"] = float((np.arange(1,11) * c).sum() / c.sum())
    return s