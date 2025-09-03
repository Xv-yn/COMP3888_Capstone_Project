import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def generate_pkl_from_coco(json_path, class_map, img_root, save_path = None, time_steps=1, return_data=False):
    """
    Convert COCO JSON annotations into .pkl dataset for training.

    Args:
        json_path (str): Path to COCO-style JSON with keypoints.
        class_map (dict): Mapping behavior name -> label index.
                         e.g. {"feeding":0, "lying":1, "standing":2, "walking":3}
        save_path (str): Where to save the .pkl file.
        time_steps (int): Sequence length (default=1).

    """
    with open(json_path, "r") as f:
        coco = json.load(f)
    
    # Map image_id -> filename, width, height
    id_to_img = {
        img["id"]: (img["file_name"], img["width"], img["height"])
        for img in coco["images"]
    }
    
    features, labels = [] , []
    
    for ann in coco["annotations"]:
        img_file, W, H = id_to_img[ann["image_id"]]
        keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)

        # normalize by image size
        keypoints[:, 0] /= W
        keypoints[:, 1] /= H

        # (T=1, V=16, C=3)
        fts = np.expand_dims(keypoints, axis=0)

        # pad to (T, V, C) if needed
        if time_steps > 1:
            pad = np.zeros((time_steps-1, keypoints.shape[0], 3), dtype=np.float32)
            fts = np.concatenate([fts, pad], axis=0)

        # assign label from file path
        label_idx = None
        for cname, idx in class_map.items():
            if os.path.exists(os.path.join(img_root, cname, img_file)):
                label_idx = idx
                break
        if label_idx is None:
            continue  # skip if not found

        features.append(fts)
        labels.append(label_idx)

    features = np.stack(features)   # (N, T, V, C)
    labels = np.array(labels)
    labels_onehot = np.eye(len(class_map))[labels]
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump((features, labels_onehot), f)
        print(f"Saved {save_path}, features={features.shape}, labels={labels_onehot.shape}")

    if return_data:
        return features, labels_onehot

def generate_train_val_from_coco(json_path, class_map, img_root, save_train, save_val, test_size=0.2, time_steps=1):
    # Load everything first
    features, labels = generate_pkl_from_coco(
        json_path=json_path,
        class_map=class_map,
        img_root=img_root,
        save_path=None,
        time_steps=time_steps,
        return_data=True
    )

    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=np.argmax(labels, axis=1)
    )

    os.makedirs(os.path.dirname(save_train), exist_ok=True)
    with open(save_train, "wb") as f:
        pickle.dump((X_train, y_train), f)
    with open(save_val, "wb") as f:
        pickle.dump((X_val, y_val), f)

    print(f"Saved train: {X_train.shape}, {y_train.shape}")
    print(f"Saved val:   {X_val.shape}, {y_val.shape}")
