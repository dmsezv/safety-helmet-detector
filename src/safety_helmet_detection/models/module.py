"""FasterRCNN model module for PyTorch Lightning."""

import pytorch_lightning as pl
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class SafetyHelmetDetector(pl.LightningModule):
    """FasterRCNN-based detector for safety helmet detection."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build FasterRCNN with custom head."""
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if self.cfg.model.pretrained else None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
        return model

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        self.log("train_loss", loss, prog_bar=True)
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # FasterRCNN needs train mode to compute losses
        self.model.train()
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.model.eval()

        self.log("val_loss", loss, prog_bar=True)
        for key, value in loss_dict.items():
            self.log(f"val_{key}", value)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.model.lr,
            momentum=self.cfg.model.get("momentum", 0.9),
            weight_decay=self.cfg.model.get("weight_decay", 0.0005),
        )
