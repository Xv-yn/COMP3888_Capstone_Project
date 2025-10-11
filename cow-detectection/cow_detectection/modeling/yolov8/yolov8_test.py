# detect_yolov8_simple.py
from ultralytics import YOLO
import cv2

# --- Paths ---
model_path = "weights/yolov8m.pt"    # Pretrained model
image_path = "data/images/img1.jpg"   # Test image
save_path = "results/output.jpg"

# --- Load model ---
model = YOLO(model_path)

# --- Run inference ---
results = model.predict(source=image_path, save=True, project="results", name="run1")

# --- Display result (optional) ---
res_img = cv2.imread("results/run1/img1.jpg")
cv2.imshow("Detection", res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
