"""이미지 → pooler feature(.npy). 사용: python -m src.extract --backbone dinov2 --gpu 0"""
import torch, numpy as np, os, glob, re, random, argparse
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor

ROOT  = "/data2/seyeon"
DS    = f"{ROOT}/datasets"
CACHE = f"{ROOT}/myjob/aesthetic-transfer/cache"
BACKBONES = {"dinov2":"ckpt/dinov2-base", "clip":"ckpt/clip-vit-b16", "siglip2":"ckpt/siglip2-base"}

# ── 어떤 이미지를 뽑을지 결정하는 부분 ──────────────────
def _norm(p): return re.sub(r'^\d\.\d{3}_', '', os.path.basename(p))  # AADB test의 0.500_ 제거

def _index(dirs):                                     # {파일명: 실제경로}
    idx = {}
    for d in dirs:
        for p in glob.glob(f"{d}/**/*.jpg", recursive=True):
            idx.setdefault(_norm(p), p)
    return idx

def get_paths(ds, split):
    if ds == "aadb":                                  # 사진 + 11속성 (probe 학습원)
        S = {"train":"Train","test":"TestNew"}[split]
        f = f"{DS}/AADB/labels/imgListFiles_label/imgList{S}Regression_score.txt"
        names = [l.split()[0] for l in open(f) if l.strip()]
        idx = _index([f"{DS}/AADB/images_train", f"{DS}/AADB/images_test"])
    elif ds == "baid":                                # 예술 (R 도메인)
        names = __import__("pandas").read_csv(f"{DS}/BAID/dataset/{split}_set.csv")["image"].tolist()
        idx = _index([f"{DS}/BAID/images"])
    elif ds == "ava":                                 # 사진 (P 도메인)
        if split == "test":
            f = f"{DS}/AVA_raw/AVA_dataset/aesthetics_image_lists/generic_test.jpgl"
            ids = [l.strip() for l in open(f)]
            random.Random(42).shuffle(ids)            # freeze: seed=42, n=5000
            names = [f"{i}.jpg" for i in ids[:5000]]
        elif split == "train":
            f = f"{DS}/AVA_raw/AVA_dataset/aesthetics_image_lists/generic_ls_train.jpgl"
            ids = [l.strip() for l in open(f)]
            random.Random(42).shuffle(ids)            # freeze: seed=42, n=20000
            names = [f"{i}.jpg" for i in ids[:20000]]
        idx = _index([f"{DS}/AVA_raw/AVA_dataset/image"])
    kept = [n for n in names if n in idx]
    return kept, [idx[n] for n in kept]

# ── 이미지를 배치로 읽는 부분 ───────────────────────────
class ImgDS(Dataset):
    def __init__(self, paths, proc): self.paths, self.proc = paths, proc
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.proc(images=Image.open(self.paths[i]).convert("RGB"),
                         return_tensors="pt")["pixel_values"][0]

# ── 모델에 통과시켜 벡터 뽑는 부분 ──────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--jobs", default="aadb:train,aadb:test,baid:test,ava:test")  # test 우선
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu); dev = f"cuda:{a.gpu}"
    os.makedirs(CACHE, exist_ok=True)

    proc  = AutoImageProcessor.from_pretrained(f"{ROOT}/{BACKBONES[a.backbone]}")
    model = AutoModel.from_pretrained(f"{ROOT}/{BACKBONES[a.backbone]}", torch_dtype=torch.float16).to(dev).eval()
    model = model.vision_model if hasattr(model, "vision_model") else model

    for job in a.jobs.split(","):
        ds, split = job.split(":")
        out = f"{CACHE}/{a.backbone}_{ds}_{split}.npy"
        if os.path.exists(out): print("skip", out, flush=True); continue
        names, paths = get_paths(ds, split)
        print(f"{a.backbone} {ds}:{split}  {len(paths)} imgs", flush=True)
        dl = DataLoader(ImgDS(paths, proc), batch_size=a.bs, num_workers=8, pin_memory=True)
        feats = []
        with torch.no_grad():
            for i, x in enumerate(dl):
                feats.append(model(pixel_values=x.to(dev, torch.float16)).pooler_output.float().cpu().numpy())
                if i % 20 == 0: print(f"  {i*a.bs}/{len(paths)}", flush=True)
        np.save(out, np.concatenate(feats))
        open(out.replace(".npy","_names.txt"),"w").write("\n".join(names))  # ★ 순서 보존
        print("saved", out, flush=True)

if __name__ == "__main__":
    main()