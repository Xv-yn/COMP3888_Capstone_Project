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


### Running the Application

Currently, the application can be run from the command line.

```bash
cd animal_action
python tools/top_down_video_det.py ./path/to/img [--no-skeleton]
```

#### Examples
- **With skeleton overlay**
  ```bash
  python tools/top_down_video_det.py examples/img3.png
  ```

- **Without skeleton overlay**
  ```bash
  python tools/top_down_video_det.py examples/img3.png --no-skeleton
  ```

## Technologies Used
This project leverages a combination of deep learning frameworks and pose estimation libraries:  

- **[PyTorch](https://pytorch.org/)** – Core deep learning framework for model training and inference.  
- **[TensorFlow](https://www.tensorflow.org/)** – Alternative framework explored for model benchmarking.  
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** – State-of-the-art object detection and keypoint estimation backbone.  
- **[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)** – Specialized toolkit for animal pose estimation, used for comparison and fine-tuned training.  

Together, these tools allow us to:  
- Detect cattle in images.  
- Estimate body keypoints (skeleton overlays).  
- Map poses into meaningful behaviours (lying, standing, walking, feeding).  

##  Datasets
We use public cattle pose datasets for training and evaluation:
- [CattlePoseEstimationDataset](https://huggingface.co/datasets/gtsaidata/CattlePoseEstimationDataset)  
- [CattleEyeView](https://github.com/AnimalEyeQ/CattleEyeView)  
- [Cattle Side Pose (Roboflow)](https://universe.roboflow.com/shi-wei-hao/cattle_side_pose/dataset/2)  

---    

For more information about datasets, technologies, methodology, and project details, see the [Wiki](https://github.com/Xv-yn/COMP3888_Capstone_Project/wiki).  

