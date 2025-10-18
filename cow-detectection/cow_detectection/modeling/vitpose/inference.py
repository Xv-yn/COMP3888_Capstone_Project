# vitpose_inference.py
from pathlib import Path
import json, pickle, cv2, numpy as np, torch
from torchvision import transforms
from model_vitpose import ViTPoseLike

ROOT       = Path(__file__).resolve().parent
DATA_ROOT  = ROOT / "training_set"  # <-- points to your raw images root (for schema & full-image tests)
SCHEMA_JSON = DATA_ROOT / "_annotations_train.coco.json"  # use any split that has 'categories'
INPUT_DIR  = ROOT / "data" / "input" / "cow"  # crops for Stage-2 (optional)
IMAGES_DIR = ROOT / "images"           # full frames if you do mapping
OUT_DIR    = ROOT / "results"; (OUT_DIR / "vis").mkdir(parents=True, exist_ok=True)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (256,192); HEATMAP_SIZE = (64,48)
CKPT       = ROOT / "weights" / "vitpose_transfer_best.pth"

# --- read schema from your COCO annotation (keypoint names + skeleton) ---
d = json.loads(SCHEMA_JSON.read_text())
cat = next((c for c in d["categories"] if "keypoints" in c and c["keypoints"]), None)
KEYPOINT_NAMES = cat["keypoints"]
NUM_KEYPOINTS  = len(KEYPOINT_NAMES)
SKELETON_0B = [(a-1, b-1) for a,b in cat.get("skeleton", [])]
NAME_TO_IDX = {n:i for i,n in enumerate(KEYPOINT_NAMES)}

# --- model ---
model = ViTPoseLike(num_keypoints=NUM_KEYPOINTS, img_size=IMAGE_SIZE, vit_name="vit_base_patch16_224").to(DEVICE)
model.eval()
state = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(state, strict=False)

# --- transforms ---
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def heatmap_argmax(hm):
    B,K,H,W = hm.shape
    idx = torch.argmax(hm.view(B,K,-1), 2, keepdim=True)
    xs = (idx % W).float(); ys = (idx // W).float()
    return torch.cat([xs,ys],2), torch.amax(hm, dim=(2,3))

def draw_skeleton(img_bgr, pts_xy):
    vis = img_bgr.copy()
    for (a,b) in SKELETON_0B:
        xa,ya = int(pts_xy[a,0]), int(pts_xy[a,1])
        xb,yb = int(pts_xy[b,0]), int(pts_xy[b,1])
        cv2.line(vis, (xa,ya), (xb,yb), (0,255,0), 2)
    for x,y in pts_xy.astype(int):
        cv2.circle(vis, (x,y), 3, (0,0,255), -1)
    return vis

# --- simple crop-only inference (tight crops in data/input) ---
crops = sorted([p for p in INPUT_DIR.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
print(f"[ViTPose] K={NUM_KEYPOINTS}  names={KEYPOINT_NAMES}")
print(f"Found {len(crops)} crops in {INPUT_DIR}")

results = []
for cp in crops:
    bgr = cv2.imread(str(cp))
    if bgr is None:
        print(f"[WARN] Cannot read {cp}, skipping.")
        continue

    orig_h, orig_w = bgr.shape[:2]

    # ----- preprocess to model input (256x192) -----
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = tf(rgb).unsqueeze(0).to(DEVICE)  # this includes Resize(256,192)

    with torch.no_grad():
        hm = model(x)  # [1,K,Hm,Wm]
        coords_hm, conf_t = heatmap_argmax(hm)

    # dynamic scaling: heatmap -> model input
    H_in, W_in = IMAGE_SIZE            # (256,192)
    H_hm, W_hm = hm.shape[2], hm.shape[3]
    sx_in = W_in / W_hm
    sy_in = H_in / H_hm

    # ensure numpy
    if isinstance(coords_hm, torch.Tensor):
        coords_hm = coords_hm[0].detach().cpu().numpy()
    else:
        coords_hm = coords_hm[0]
    if isinstance(conf_t, torch.Tensor):
        conf = conf_t[0].detach().cpu().numpy()
    else:
        conf = conf_t[0]

    # heatmap -> model-input coords (256x192)
    pts_in = np.stack([coords_hm[:, 0] * sx_in, coords_hm[:, 1] * sy_in], axis=1)

    # ----- model-input -> original-crop coords -----
    # inverse of the resize from (orig_h,orig_w) -> (H_in,W_in)
    sx_out = orig_w / W_in
    sy_out = orig_h / H_in
    pts_orig = np.stack([pts_in[:, 0] * sx_out, pts_in[:, 1] * sy_out], axis=1)

    # draw on the ORIGINAL crop (no resize)
    vis = draw_skeleton(bgr, pts_orig)
    cv2.imwrite(str((OUT_DIR / "vis" / cp.name)), vis)

    # save record in ORIGINAL crop coordinates
    keypoints = [
        {"id": k, "name": KEYPOINT_NAMES[k],
         "x": float(pts_orig[k, 0]), "y": float(pts_orig[k, 1]),
         "score": float(conf[k])}
        for k in range(NUM_KEYPOINTS)
    ]
    results.append({
        "image": cp.name,
        "bbox": [0, 0, float(orig_w), float(orig_h)],  # original crop extent
        "keypoints": keypoints
    })

# save once
with open(OUT_DIR / "keypoints.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"Saved {len(results)} skeletons to {OUT_DIR/'keypoints.pkl'} | visuals → {OUT_DIR/'vis'}")
