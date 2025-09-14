"""
Inference pipeline for cow behavior recognition.

Implements the CowInference class, which inherits from BaseInference.
This class orchestrates the full pipeline:
1. Detect cows in images/videos using YOLOv8.
2. Extract skeleton keypoints from cropped cow regions using HRNet.
3. Classify behaviors using the trained ST-GCN model.
Results are saved as ROI images and a CSV file with predictions and latency.
"""
