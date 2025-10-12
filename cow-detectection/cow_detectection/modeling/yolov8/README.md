# YOLOv8: Training & Inference

This repository provides two convenience scripts for Ultralytics YOLOv8:

- `train.py` — trains a detector on a dataset in standard YOLO format.

- `predict.py` — runs inference on one image or all images in a folder.

The goal is a minimal, reproducible workflow with clear outputs and sensible defaults.

## Usage

### Inference

predict.py will:

- Load weights/yolov8m.pt by default.
- Accept `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`.

```bash
# Single image
python predict.py /path/to/image.jpg

# All images in a folder
python predict.py /path/to/folder
```

Will write predictions (images with boxes) under:

```bash
yolov7/results/run/
```

### Training

The command to be run to train the model.

```bash
python train.py /path/to/dataset \
  --weights weights/yolov8m.pt \
  --epochs 200 \
  --batch 16 \
  --imgsz 640 \
  --name exp1
```

###### Memory tips (GPU OOM)

If CUDA out of memory occurs:

- Reduce batch size, e.g. --batch 8 or --batch 4.
- Use a smaller model, e.g. --weights yolov8n.pt or yolov8s.pt.
- Reduce image size, e.g. --imgsz 512.

## Dataset Layout (YOLO format)

This is the expected format of the training data.

```
<dataset_root>/
  ├─ train/
  │   ├─ images/*.jpg|png|bmp|tif|tiff
  │   └─ labels/*.txt            # one file per image; "class cx cy w h" per line
  ├─ val/
  │   ├─ images/*.jpg|png|bmp|tif|tiff
  │   └─ labels/*.txt
  └─ data.yaml  (optional)
```

## Importing run_inference from another module

If the package layout includes **init**.py files, run_inference can be imported
directly:

```python
from ultralytics import YOLO
from cow_detectection.modeling.yolov8.predict import run_inference

model = YOLO("weights/yolov8m.pt")
output = run_inference(model, "/path/to/image.jpg")
```

> [!NOTE]
> If output_dir is empty ("" or None), it won’t save images; otherwise it saves
> under the specified directory.
