"""
Trainer for the ST-GCN model.

Implements the CowTrainer class, which inherits from BaseTrainer and handles
the training loop for the ST-GCN model. This includes optimizer setup,
loss calculation, validation, checkpoint saving, and optional CLI support
for launching training runs.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
import typer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from cow_detectection.modeling.base import BaseTrainer
from cow_detectection.modeling.stgcn.model import StreamSpatialTemporalGraph, TwoStreamSpatialTemporalGraph
from cow_detectection.modeling.stgcn.preprocessor import KeypointPreprocessor
from cow_detectection.modeling.stgcn.utils import load_dataset


class CowTrainer(BaseTrainer):
    """
    Trainer class for ST-GCN models.

    Inherits from BaseTrainer and provides:
    - Initialization of model, optimizer, and loss function.
    - Training loop with optional validation.
    - Checkpoint saving/loading.
    - Metric logging.
    """
    def __init__(self,
                 model: nn.Module,
                 device: str = "cpu",
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 criterion: nn.Module = None,
                 optimizer_cls: type = optim.Adam,
                 checkpoint_dir: str = "models/checkpoints"
     ):
        super().__init__(model)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.checkpoint_dir = checkpoint_dir


    def train(self, train_loader, val_loader=None, num_epochs=30, log_interval=10):
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            for phase in ["train", "val" if val_loader else "train"]:
                if phase == "train":
                    self.model.train()
                    dataloader = train_loader
                else:
                    self.model.eval()
                    dataloader = val_loader

                running_loss, running_acc = 0.0, 0.0

                for batch_idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == labels).float().mean().item()

                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item()
                    running_acc += acc

                    if batch_idx % log_interval == 0:
                        logger.info(f"{phase} | Batch {batch_idx} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = running_acc / len(dataloader)
                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc)

                logger.info(f"📊 {phase.capitalize()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch)
        
        return history


    def save_checkpoint(self, epoch, filename="checkpoint.pth"):
        path = f"{self.checkpoint_dir}/epoch_{epoch}_{filename}"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info(f"💾 Checkpoint saved at {path}")


    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"✅ Loaded checkpoint from {path}")

# CLI entrypoint with Typer
app = typer.Typer()

@app.command()
def main(
    model_type: str = typer.Option("onestream", help="Model type: onestream or twostream"),
    epochs: int = 10,
    batch_size: int = 32,
    device: str = "cpu"
):
    """
    CLI entrypoint for training the ST-GCN model.
    Example:
        python cow_detectection/modeling/stgcn/trainer.py --epochs 20 --batch-size 64 --device cuda
    """

    # TODO: initialize dataset, dataloaders, model, trainer
    # 1. Load and preprocess data
    df = load_dataset(path = 'None')
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    preprocessor = KeypointPreprocessor(two_stream=(model_type == "twostream"))
    # 2. Load train/test dataframes
    preprocessor.get_data_and_labels(df_train, df_test)

    # 3. Scale features
    X_train_scaled, X_val_scaled = preprocessor.scale(
        StandardScaler(),
        preprocessor.X_train_,
        preprocessor.X_test_
    )

    # 4. Build datasets
    train_data = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(preprocessor.y_train_, dtype=torch.long)
    )
    val_data = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(preprocessor.y_test_, dtype=torch.long)
    )

    # 5. Build dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # 6. Build model
    graph_args = {"layout": "coco_cut", "strategy": "spatial"}
    if model_type == "onestream":
        model = StreamSpatialTemporalGraph(3, graph_args, num_class=3)
    elif model_type == "twostream":
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class=3)
    else:
        raise ValueError("Invalid model_type")

    # 7. Train model
    trainer = CowTrainer(model, device=device)
    trainer.train(train_loader, val_loader, num_epochs=epochs)
    logger.info("Training entrypoint not yet implemented.")


if __name__ == "__main__":
    app()
