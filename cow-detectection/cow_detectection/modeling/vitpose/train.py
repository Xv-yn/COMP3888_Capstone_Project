# train_vitpose.py
from pathlib import Path
import time, torch, torch.nn as nn
from torch.utils.data import DataLoader

from model_vitpose import ViTPoseLike
from dataset_cattle import CattleKeypointDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- point these at your data ----
DATA_ROOT   = Path("./training_set")
TRAIN_JSON  = DATA_ROOT / "_annotations_train.coco.json"
VAL_JSON    = DATA_ROOT / "_annotations_valid.coco.json"
IMAGES_DIR  = DATA_ROOT                  # we search recursively under this
WEIGHTS_DIR = Path("./weights"); WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_OUT    = WEIGHTS_DIR / "vitpose_transfer_best.pth"

IMAGE_SIZE  = (256, 192)   # (H, W)
HEATMAP_SZ  = (64, 48)
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 5e-4
WD          = 1e-4

def build_loaders():
    train_ds = CattleKeypointDataset(IMAGES_DIR, TRAIN_JSON, input_hw=IMAGE_SIZE, heatmap_hw=HEATMAP_SZ, is_train=True)
    val_ds   = CattleKeypointDataset(IMAGES_DIR, VAL_JSON,   input_hw=IMAGE_SIZE, heatmap_hw=HEATMAP_SZ, is_train=False)
    schema   = train_ds.get_schema()
    print(f"[schema] K={schema['num_keypoints']}  names={schema['keypoint_names']}")
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)    
    return train_ld, val_ld, schema

def train_one_epoch(model, loader, opt, crit):
    model.train(); total = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE); targets = targets.to(DEVICE)
        preds = model(imgs)
        loss = crit(preds, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); total = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE); targets = targets.to(DEVICE)
        preds = model(imgs)
        loss = crit(preds, targets)
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

def main():
    train_ld, val_ld, schema = build_loaders()
    K = schema["num_keypoints"]
    model = ViTPoseLike(num_keypoints=K, img_size=IMAGE_SIZE, vit_name="vit_base_patch16_224").to(DEVICE)

    # freeze backbone → train neck+head
    for p in model.backbone.parameters(): p.requires_grad = False
    params = list(model.neck.parameters()) + list(model.head.parameters())
    opt  = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    crit = nn.MSELoss()

    best = 1e9
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr = train_one_epoch(model, train_ld, opt, crit)
        val = evaluate(model, val_ld, crit)
        print(f"Epoch {epoch:02d}: train {tr:.4f} | val {val:.4f} | {time.time()-t0:.1f}s")
        if val < best:
            best = val
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"  ✓ saved {CKPT_OUT}")

    print("Done. Best val:", best)

if __name__ == "__main__":
    main()
