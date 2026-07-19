"""ArtiMuse (InternVL3-8B) aesthetic scoring on AVA_test + BAID_test → SRCC.
사용: conda activate artimuse; python -m src.score_artimuse"""
import torch, numpy as np, scipy.stats, os, re
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from src.data import baid_scores, ava_scores

CACHE = "/data2/seyeon/myjob/aesthetic-transfer/cache"
DS = "/data2/seyeon/datasets"
CKPT = "/data2/seyeon/ckpt/artimuse"

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)
tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                T.Resize((448,448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(), T.Normalize(MEAN,STD)])
PROMPT = "<image>\nRate the aesthetic quality of this image on a 0-100 scale. Output only one number."
GEN = dict(max_new_tokens=16, do_sample=False)

def img_path(domain, name):
    return f"{DS}/BAID/images/{name}" if domain=="baid" else f"{DS}/AVA_raw/AVA_dataset/image/{name}"

def names_of(domain):
    return [n for n in open(f"{CACHE}/dinov2_{domain}_test_names.txt").read().split("\n") if n]

def parse_score(resp):
    m = re.search(r'-?\d+\.?\d*', str(resp))
    return float(m.group()) if m else None

def score_domain(model, tok, domain, dev, bs=8):
    sc = baid_scores("test") if domain=="baid" else ava_scores()
    names = [n for n in names_of(domain) if n in sc]
    preds, gts = [], []
    buf_pv, buf_n = [], []

    def flush():
        if not buf_pv: return
        pv = torch.stack(buf_pv).to(torch.float16).to(dev)
        qs = [PROMPT]*len(buf_pv)
        npl = [1]*len(buf_pv)   # 각 이미지 1패치 (fixed res)
        resps = model.batch_chat(tok, pv, qs, GEN, num_patches_list=npl)
        for n, r in zip(buf_n, resps):
            s = parse_score(r)
            if s is not None:
                preds.append(s); gts.append(sc[n])
        buf_pv.clear(); buf_n.clear()

    for i, n in enumerate(names):
        try:
            img = Image.open(img_path(domain, n))
            buf_pv.append(tf(img)); buf_n.append(n)
        except Exception:
            continue
        if len(buf_pv) >= bs: flush()
        if i % 200 == 0: print(f"  {domain} {i}/{len(names)}  (kept {len(preds)})", flush=True)
    flush()

    srcc = scipy.stats.spearmanr(preds, gts).correlation
    plcc = scipy.stats.pearsonr(preds, gts)[0]
    np.save(f"{CACHE}/artimuse_{domain}_scores.npy", np.array(preds))
    print(f"ArtiMuse {domain}: n={len(preds)} SRCC={srcc:.4f} PLCC={plcc:.4f}", flush=True)
    return srcc

if __name__ == "__main__":
    dev = "cuda:0"
    m = AutoModel.from_pretrained(CKPT, trust_remote_code=True, torch_dtype=torch.float16).eval().to(dev)
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True, use_fast=False)
    print("=== ArtiMuse scoring ===")
    r_ava  = score_domain(m, tok, "ava", dev)
    r_baid = score_domain(m, tok, "baid", dev)
    print(f"\nArtiMuse  photo(AVA)={r_ava:.4f}  art(BAID)={r_baid:.4f}  drop={r_ava-r_baid:.4f}")