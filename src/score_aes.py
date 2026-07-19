"""Aesthetic Predictor V2.5 scoring on AVA_test + BAID_test → SRCC.
사용: (advanced_ai) python -m src.score_aes"""
import torch, numpy as np, scipy.stats, os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from src.data import baid_scores, ava_scores

CACHE = "/data2/seyeon/myjob/aesthetic-transfer/cache"
DS = "/data2/seyeon/datasets"

def img_path(domain, name):
    return f"{DS}/BAID/images/{name}" if domain=="baid" else f"{DS}/AVA_raw/AVA_dataset/image/{name}"

def names_of(domain):
    return [n for n in open(f"{CACHE}/dinov2_{domain}_test_names.txt").read().split("\n") if n]

def score_domain(model, preprocessor, domain, dev, bs=32):
    sc = baid_scores("test") if domain=="baid" else ava_scores()
    names = [n for n in names_of(domain) if n in sc]
    preds, gts, batch, bnames = [], [], [], []

    def flush():
        if not batch: return
        px = preprocessor(images=batch, return_tensors="pt").pixel_values.to(dev, torch.float16)
        with torch.no_grad():
            s = model(px).logits.squeeze(-1).float().cpu().numpy()
        preds.extend(s.tolist()); gts.extend([sc[n] for n in bnames])
        batch.clear(); bnames.clear()

    for i, n in enumerate(names):
        try:
            batch.append(Image.open(img_path(domain,n)).convert("RGB")); bnames.append(n)
        except Exception:
            continue
        if len(batch) >= bs: flush()
        if i % 1000 == 0: print(f"  {domain} {i}/{len(names)}", flush=True)
    flush()

    srcc = scipy.stats.spearmanr(preds, gts).correlation
    plcc = scipy.stats.pearsonr(preds, gts)[0]
    np.save(f"{CACHE}/aes_{domain}_scores.npy", np.array(preds))
    print(f"Aes-V2.5 {domain}: n={len(preds)} SRCC={srcc:.4f} PLCC={plcc:.4f}", flush=True)
    return srcc

if __name__ == "__main__":
    dev = "cuda:3"    # Q-Align이 cuda:0 쓰는 중이면 다른 GPU
    model, preprocessor = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(dev, torch.float16).eval()
    print("=== Aes-V2.5 scoring ===")
    r_ava  = score_domain(model, preprocessor, "ava", dev)
    r_baid = score_domain(model, preprocessor, "baid", dev)
    print(f"\nAes-V2.5  photo(AVA)={r_ava:.4f}  art(BAID)={r_baid:.4f}  drop={r_ava-r_baid:.4f}")