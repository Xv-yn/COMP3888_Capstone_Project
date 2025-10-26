# cow-detectection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Application for farmer to track welfare of cow in a farm

## Project Organization

```
 ├── .git/                      <- Git repository files for version control
 │
 ├── cow-detectection/          <- Main project directory (legacy or alternate version)
 │
 │   ├── cow_detectection/          <- Primary source code package for the cow detection system
 │   │   ├── __pycache__/           <- Compiled Python bytecode
 │   │   │
 │   │   ├── modeling/              <- Core model definitions and inference scripts
 │   │   │   ├── __pycache__/
 │   │   │   │
 │   │   │   ├── hrnet/             <- HRNet model for keypoint or pose estimation
 │   │   │   │   ├── config/        <- Configuration files for HRNet
 │   │   │   │   ├── weights/       <- Pretrained HRNet model weights and modules
 │   │   │   │   │   ├── __init__.py
 │   │   │   │   │   ├── hrnet_inference.py
 │   │   │   │   │   ├── pose_inference.py
 │   │   │   │   │   ├── pose_utils.py
 │   │   │   │   │   ├── hrnet.png
 │   │   │   │   │   └── README.md
 │   │   │   │
 │   │   │   ├── mmpose/            <- Placeholder or optional module for MMPose-based models
 │   │   │   │
 │   │   │   ├── stgcn/             <- Spatial-Temporal Graph Convolutional Network for action recognition
 │   │   │   │   ├── __pycache__/
 │   │   │   │   ├── weights/       <- Pretrained ST-GCN weights and code
 │   │   │   │   │   ├── __init__.py
 │   │   │   │   │   ├── model.py
 │   │   │   │   │   ├── predict.py
 │   │   │   │   │   ├── preprocessor.py
 │   │   │   │   │   ├── train.py
 │   │   │   │   │   ├── STGCN.png
 │   │   │   │   │   └── README.md
 │   │   │   │   └── utils.py       <- Utility functions for ST-GCN
 │   │   │   │
 │   │   │   ├── yolov8/            <- YOLOv8-based object detection pipeline
 │   │   │   │   ├── __pycache__/
 │   │   │   │   ├── data/          <- Datasets or data configs used for YOLOv8
 │   │   │   │   ├── results/       <- Model outputs, detections, and evaluation results
 │   │   │   │   ├── weights/       <- YOLOv8 model weights and scripts
 │   │   │   │   │   ├── __init__.py
 │   │   │   │   │   ├── plots.py
 │   │   │   │   │   ├── predict.py
 │   │   │   │   │   ├── train.py
 │   │   │   │   │   ├── yolov8_inference.py
 │   │   │   │   │   ├── YOLOv8.png
 │   │   │   │   │   └── README.md
 │   │   │   │   ├── yolov8_test.py <- Test scripts for YOLOv8 model
 │   │   │   │   ├── base.py        <- Shared base model or inference class
 │   │   │   │   ├── predict.py     <- Unified prediction interface
 │   │   │   │   ├── img3.png
 │   │   │   │   ├── Overall.png
 │   │   │   │   └── __init__.py
 │   │   │   │
 │   │   │   └── README.md          <- Documentation for modeling subpackage
 │   │   │
 │   │   ├── results/               <- Directory to store inference outputs or evaluation reports
 │   │   │
 │   │   └── __init__.py            <- Makes cow_detectection a Python package
 │   │
 │   ├── docs/                      <- Documentation and project metadata
 │   │   ├── .gitignore
 │   │   ├── LICENSE                <- License file for the project
 │   │   ├── Makefile               <- Build automation and convenience commands
 │   │   ├── pyproject.toml         <- Python project configuration (build, packaging, metadata)
 │   │   ├── README.md              <- Primary documentation for project usage
 │   │   ├── requirements.txt       <- Python dependencies for reproducibility
 │   │   └── run_inference.sh       <- Shell script for running inference pipeline
 │   │
 │   ├── Document/                  <- Supporting documents or research materials
 │   │
 │   ├── training_set/              <- Dataset or images used for model training
 │   │
 │   ├── venv/                      <- Python virtual environment for project dependencies
 │   │   └── README.md
 │   │
 │   └── setup.cfg                  <- Configuration for linting, formatting, and packaging tools
```
