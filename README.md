# AI Models for Livestock Pose Estimation

This project applies **deep learning** and **computer vision** to automatically detect livestock (cattle) poses and classify behaviours (e.g., standing, lying, walking, feeding).  
The goal is to create a **usable, accurate, and efficient system** that supports livestock health monitoring and contributes to **precision farming**.

---

## Deployment

### Hardware Requirements

- GPU recommended (≥8GB VRAM for training)
- CPU (4 cores or more)
- RAM ≥16GB
- Disk space ≥10GB

### Software Requirements

- Python **3.10**
- Virtual environment (`venv` recommended)
- Dependencies listed in `requirements.txt`

### Setting up a Virtual Environment

#### Using Python venv

1. **Clone the repository**

   ```bash
   git clone https://github.com/Xv-yn/COMP3888_Capstone_Project
   cd COMP3888_Capstone_Project
   ```

2. **Create the virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the environment**
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```bash
     .\venv\Scripts\Activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. Your environment is now ready. ✅

#### Option 2: Using Conda (via Makefile)

If you prefer using Conda, the Makefile already provides a setup command:

```bash
make create_environment
```

This will create a Conda environment named `cow-detectection` using Python 3.10.  
Activate it after creation:

```bash
conda activate cow-detectection
```

Then install dependencies:

```bash
make requirements
```

### Downloading weights

Below is a link to all the weights.

https://drive.google.com/drive/folders/18MqaMUk8Lhn4EubUblOTmskwcoxm7LUM?usp=sharing

Place each of the weights in their respective folders as seen below:

```
 └── cow-detectection/
     └── cow_detectection/
         └── modeling/
             ├── hrnet/
             │   └── weights/
             │       └── hrnet_w32_ap10k.pth
             ├── stgcn/
             │   └── weights/
             │       └── tsstg-model.pth
             └── yolov8/
                 └── weights/
                     └── yolov8m.pt
```

### Running the Application

Currently, the application can be run from the command line.

```bash
cd cow-detectection
./run_inference.sh 3 ./cow_detectection/modeling/img3.png
```

#### Customized Examples

- **With skeleton overlay**

  ```bash
  cd cow-detectection/cow_detectection/modeling
  python predict.py --option 3 --image-path /path/to/image.jpg --device cuda --show-skeleton
  ```

- **Without skeleton overlay**
  ```bash
  python predict.py --option 3 --image-path /path/to/image.jpg --device cuda --no-show-skeleton
  ```

## Technologies Used

This project leverages a combination of deep learning frameworks and pose estimation libraries:

- **[PyTorch](https://pytorch.org/)** – Core deep learning framework for model training and inference.
- **[TensorFlow](https://www.tensorflow.org/)** – Alternative framework explored for model benchmarking.
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** – State-of-the-art object detection and keypoint estimation backbone.
- **[HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation)** – For high-precision keypoint/skeleton detection.
- **[ST-GCN (TS-STG)](https://github.com/yysijie/st-gcn)** – For spatio-temporal graph convolution-based action recognition.

Together, these tools allow us to:

- Detect cattle in images.
- Estimate body keypoints (skeleton overlays).
- Map poses into meaningful behaviours (lying, standing, walking, feeding).

## Datasets

We use public cattle pose datasets for training and evaluation:

- [CattlePoseEstimationDataset](https://huggingface.co/datasets/gtsaidata/CattlePoseEstimationDataset)
- [CattleEyeView](https://github.com/AnimalEyeQ/CattleEyeView)
- [Cattle Side Pose (Roboflow)](https://universe.roboflow.com/shi-wei-hao/cattle_side_pose/dataset/2)

---

For more information about datasets, technologies, methodology, and project details, see the [Wiki](https://github.com/Xv-yn/COMP3888_Capstone_Project/wiki).
