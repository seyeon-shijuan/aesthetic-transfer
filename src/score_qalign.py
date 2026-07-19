"""Q-Align aesthetic scoring on AVA_test + BAID_test → SRCC.
사용: conda activate qalign; python -m src.score_qalign"""
import torch, numpy as np, pandas as pd, scipy.stats, os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import AutoModelForCausalLM
from src.data import baid_scores, ava_scores
from src.probe import load_feat   # names 재사용 (feature 추출 때 저장한 순서)

CACHE = "/data2/seyeon/myjob/aesthetic-transfer/cache"
DS = "/data2/seyeon/datasets"

def img_path(domain, name):
    if domain == "baid":
        return f"{DS}/BAID/images/{name}"
    else:  # ava
        return f"{DS}/AVA_raw/AVA_dataset/image/{name}"

def score_domain(model, domain):
    # feature 추출 때 저장한 names 순서 재사용 → 동일 이미지 집합
    _, names = load_feat("dinov2", domain, "test")
    sc = baid_scores("test") if domain == "baid" else ava_scores()
    names = [n for n in names if n in sc]

    preds, gts = [], []
    for i, n in enumerate(names):
        try:
            img = Image.open(img_path(domain, n)).convert("RGB")
            s = float(model.score([img], task_="aesthetics", input_="image"))
        except Exception as e:
            continue
        preds.append(s); gts.append(sc[n])
        if i % 500 == 0:
            print(f"  {domain} {i}/{len(names)}", flush=True)

    srcc = scipy.stats.spearmanr(preds, gts).correlation
    plcc = scipy.stats.pearsonr(preds, gts)[0]
    np.save(f"{CACHE}/qalign_{domain}_scores.npy", np.array(preds))
    print(f"Q-Align {domain}: n={len(preds)} SRCC={srcc:.4f} PLCC={plcc:.4f}", flush=True)
    return srcc

if __name__ == "__main__":
    m = AutoModelForCausalLM.from_pretrained(
        "/data2/seyeon/ckpt/q-align", trust_remote_code=True,
        attn_implementation="eager", torch_dtype=torch.float16, device_map="cuda:0")
    print("=== Q-Align scoring ===")
    r_ava  = score_domain(m, "ava")    # photo (in-domain for photo-specialist)
    r_baid = score_domain(m, "baid")   # art  (out-domain)
    print(f"\nQ-Align  photo(AVA)={r_ava:.4f}  art(BAID)={r_baid:.4f}  drop={r_ava-r_baid:.4f}")