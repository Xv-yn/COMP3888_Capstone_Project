"""
ST-GCN trainer utilities for cow action recognition (transfer-learning ready).

This module defines `CowTrainer`, a lightweight training wrapper around
(single- or two-stream) Spatial-Temporal GCN models. It handles:

    - Model / optimizer / loss initialization
    - Optional pretrained weight loading (transfer learning)
    - Train / validation loops with accuracy tracking
    - Periodic checkpoint saving
    - Typer CLI for launching training runs

Quick start:
    # One-stream ST-GCN fine-tune on CPU
    python cow_detectection/modeling/stgcn/trainer.py --epochs 10 --batch-size 32 --device cpu

    # Two-stream ST-GCN fine-tune from pretrained weights on CUDA
    python cow_detectection/modeling/stgcn/trainer.py --model-type twostream --epochs 20 --device cuda --pretrained ./weights/best.pt --freeze-backbone
"""

from loguru import logger
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import typer
from tqdm import tqdm
from cow_detectection.modeling.base import BaseTrainer
from cow_detectection.modeling.stgcn.model import (
    StreamSpatialTemporalGraph,
    TwoStreamSpatialTemporalGraph,
)
from cow_detectection.modeling.stgcn.preprocessor import KeypointPreprocessor
from pathlib import Path
import matplotlib.pyplot as plt


# ------------------------------
# Global Config
# ------------------------------
data_files = ['./data/train.pkl', './data/val.pkl']
class_names = ['Standing', 'Lying', 'Walking', 'Feeding']
num_class = len(class_names)


# ------------------------------
# CowTrainer Definition
# ------------------------------
class CowTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        criterion: nn.Module = None,
        optimizer_cls: type = optim.Adadelta,
        checkpoint_dir: str = './results/',
        save_checkpoint: str = './weights/'
    ):
        super().__init__(model)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion or nn.BCELoss()
        self.optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),  # freeze-safe
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # store dirs (avoid shadowing the save_checkpoint method)
        self.checkpoint_dir = checkpoint_dir
        self.weights_dir = save_checkpoint

        # create directories if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.weights_dir).mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Transfer Learning Loader
    # --------------------------
    def load_pretrained(self, weight_path: str, freeze_backbone: bool = False, replace_fc: bool = True):
        """Load pretrained weights and optionally freeze backbone for fine-tuning."""
        ckpt = torch.load(weight_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ Loaded pretrained weights from {weight_path}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

        # Freeze all except classifier if requested
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad = False
            logger.info("🧊 Backbone frozen for fine-tuning.")

        # Replace classifier layer if number of output classes differs
        if replace_fc:
            in_features = list(self.model.fc.parameters())[0].shape[1] if hasattr(self.model, "fc") else None
            if in_features:
                self.model.fc = nn.Linear(in_features, num_class).to(self.device)
                logger.info(f"🔁 Classifier head replaced with new output layer for {num_class} classes.")

    # --------------------------
    # Training Loop
    # --------------------------
    def train(self, train_loader, val_loader=None, num_epochs=30, log_interval=10):
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        dataloaders = {"train": train_loader, "val": val_loader}

        best_acc = 0.0
        best_path = None

        for epoch in range(num_epochs):
            logger.info(f"\n🚀 Epoch {epoch+1}/{num_epochs}")

            for phase in ["train", "val" if val_loader else "train"]:
                self.model.train() if phase == "train" else self.model.eval()
                dataloader = dataloaders[phase]
                if dataloader is None:
                    continue

                running_loss, running_acc = 0.0, 0.0

                with tqdm(dataloader, desc=f"{phase} (epoch {epoch+1})", unit="batch") as iterator:
                    for pts, lbs in iterator:
                        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
                        mot, pts, lbs = mot.to(self.device), pts.to(self.device), lbs.to(self.device)

                        with torch.set_grad_enabled(phase == "train"):
                            # forward
                            if isinstance(self.model, tuple) or hasattr(self.model, "twostream"):
                                outputs = self.model((pts, mot))
                            else:
                                outputs = self.model(pts)

                            loss = self.criterion(outputs, lbs)
                            lbs_indices = torch.argmax(lbs, dim=1)
                            preds = torch.argmax(outputs, dim=1)
                            acc = (preds == lbs_indices).float().mean().item()

                            if phase == "train":
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()

                        running_loss += loss.item()
                        running_acc += acc
                        iterator.set_postfix_str(f"loss: {loss.item():.4f}, acc: {acc:.4f}")

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = running_acc / len(dataloader)
                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc)

                logger.info(f"📊 {phase.capitalize()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

                # --- save best checkpoint based on validation accuracy ---
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_path = f"{self.checkpoint_dir}/best.pt"
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_acc": best_acc,
                        },
                        best_path,
                    )
                    logger.info(f"🏆 New best model saved with acc={best_acc:.4f} at {best_path}")

            # Optional: save periodic backup
            self.save_checkpoint(epoch)

        # --- after training, plot accuracy ---
        self.plot_accuracy(history)
        logger.info(f"✅ Training done. Best model: {best_path} (acc={best_acc:.4f})")

        return history

    def plot_accuracy(self, history):
        """Plot train/val accuracy across epochs and save as PNG."""
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_acc"], label="Train Acc")
        if len(history["val_acc"]) > 0:
            plt.plot(history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.checkpoint_dir}/accuracy_curve.png")
        plt.close()
        logger.info(f"📈 Accuracy plot saved to {self.checkpoint_dir}/accuracy_curve.png")
    # --------------------------
    # Checkpoint Helpers
    # --------------------------
    def save_checkpoint(self, epoch, filename="checkpoint.pth"):
        path = f"{self.checkpoint_dir}/epoch_{epoch}_{filename}"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"💾 Checkpoint saved at {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"✅ Loaded checkpoint from {path}")


# ------------------------------
# CLI Entrypoint
# ------------------------------
app = typer.Typer()

@app.command()
def main(
    model_type: str = typer.Option("onestream", help="Model type: onestream or twostream"),
    epochs: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
    pretrained: str = typer.Option(None, help="Optional path to pretrained weights (.pt/.pth)"),
    freeze_backbone: bool = typer.Option(False, help="Freeze backbone during fine-tuning"),
):
    """CLI entrypoint for training or fine-tuning ST-GCN on cow posture data."""

    # Load datasets
    train_loader, _ = KeypointPreprocessor.load_dataset(data_files[0:1], batch_size)
    val_loader, train_loader_ = KeypointPreprocessor.load_dataset(data_files[1:2], batch_size, 0.1)
    train_loader = data.DataLoader(
        data.ConcatDataset([train_loader.dataset, train_loader_.dataset]),
        batch_size, shuffle=True
    )
    del train_loader_

    # Build model
    graph_args = {"layout": "coco_cut", "strategy": "spatial"}
    if model_type == "onestream":
        model = StreamSpatialTemporalGraph(3, graph_args, num_class=num_class)
    elif model_type == "twostream":
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class=num_class)
    else:
        raise ValueError("Invalid model_type")

    # Init trainer
    trainer = CowTrainer(model, device=device, criterion=nn.BCELoss(), optimizer_cls=optim.Adam)

    # Load pretrained weights if provided
    if pretrained:
        trainer.load_pretrained(pretrained, freeze_backbone=freeze_backbone, replace_fc=True)

    # Train / Fine-tune
    trainer.train(train_loader, val_loader, num_epochs=epochs)
    logger.info("✅ Training completed.")


if __name__ == "__main__":
    app()
