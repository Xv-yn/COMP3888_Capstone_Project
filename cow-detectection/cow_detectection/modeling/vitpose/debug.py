# debug_one.py (with output rescaled to original image size)
from pathlib import Path
import json, cv2, numpy as np, torch
from torchvision import transforms
from model_vitpose import ViTPoseLike

ROOT = Path(__file__).resolve().parent
DATA_ROOT  = ROOT / "training_set"
SCHEMA_JSON = DATA_ROOT / "_annotations_train.coco.json"
CKPT = ROOT / "weights" / "vitpose_transfer_best.pth"
IMAGE_SIZE = (256,192)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- read schema ---
d = json.loads(SCHEMA_JSON.read_text())
cat = next(c for c in d["categories"] if "keypoints" in c and c["keypoints"])
KP_NAMES = cat["keypoints"]; K = len(KP_NAMES)

# --- load model ---
assert CKPT.exists(), f"Missing {CKPT}"
model = ViTPoseLike(num_keypoints=K, img_size=IMAGE_SIZE, vit_name="vit_base_patch16_224").to(DEVICE).eval()
state = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(state, strict=False)

# --- preprocessing ---
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def argmax_hm(hm):
    B,K,H,W = hm.shape
    idx = torch.argmax(hm.view(B,K,-1), 2, keepdim=True)
    xs = (idx % W).float(); ys = (idx // W).float()
    return torch.cat([xs,ys],2), torch.amax(hm, dim=(2,3))

def viz_heatmaps(hm):  # [K,Hm,Wm] → grid image
    import math
    hm = hm.cpu().numpy()
    k,h,w = hm.shape
    cols = int(math.ceil(math.sqrt(k))); rows = int(math.ceil(k/cols))
    cell = 96
    canvas = np.zeros((rows*cell, cols*cell), np.uint8)
    for i in range(k):
        r, c = i // cols, i % cols
        hmi = hm[i]; hmi = (hmi - hmi.min()) / (hmi.ptp()+1e-6)
        hmi = (hmi*255).astype(np.uint8)
        hmi = cv2.resize(hmi, (cell, cell), interpolation=cv2.INTER_NEAREST)
        canvas[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = hmi
    return cv2.applyColorMap(canvas, cv2.COLORMAP_JET)

# --- pick one crop ---
inp_dir = ROOT / "data" / "input" / "cow"
crops = sorted([p for p in inp_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
assert crops, f"No crops in {inp_dir}"
cp = crops[0]
bgr = cv2.imread(str(cp)); assert bgr is not None, cp
orig_h, orig_w = bgr.shape[:2]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
x = tf(rgb).unsqueeze(0).to(DEVICE)

# --- inference ---
with torch.no_grad():
    hm = model(x)  # [1,K,Hm,Wm]
    coords_hm, conf_t = argmax_hm(hm)

coords_hm = coords_hm[0].cpu().numpy()
conf = conf_t[0].cpu().numpy()

# --- dynamic scaling ---
H_in, W_in = IMAGE_SIZE
H_hm, W_hm = hm.shape[2], hm.shape[3]
sx_in = W_in / W_hm
sy_in = H_in / H_hm
pts_in = np.stack([coords_hm[:,0]*sx_in, coords_hm[:,1]*sy_in], 1)

# --- rescale back to original crop size ---
sx_out = orig_w / W_in
sy_out = orig_h / H_in
pts_orig = np.stack([pts_in[:,0]*sx_out, pts_in[:,1]*sy_out], 1)

# --- draw only dots + indices on ORIGINAL image ---
vis = bgr.copy()
for i,(xk,yk) in enumerate(pts_orig.astype(int)):
    color = (0,0,255) if conf[i] >= 0.15 else (80,80,80)
    cv2.circle(vis, (xk,yk), 3, color, -1)
    cv2.putText(vis, str(i), (xk+4,yk-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

# --- save debug outputs ---
out_vis_dir = ROOT/"results"/"vis"
out_vis_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_vis_dir/f"DEBUG_{cp.name}"), vis)
cv2.imwrite(str(out_vis_dir/f"DEBUG_HM_{cp.stem}.jpg"), viz_heatmaps(hm[0]))

# --- print summary ---
print(f"image: {cp.name}  original: {orig_w}x{orig_h}")
print("max per joint:", [round(float(m),3) for m in conf])
print("mean/max conf :", float(conf.mean()), float(conf.max()))
print("saved:", out_vis_dir/f"DEBUG_{cp.name}")
