import pytorch_lightning as pl
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class SafetyHelmetDetector(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes

        # Load model
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if cfg.model.pretrained else None)

        # Replace head for our classes (background + classes)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        # images is list of tensors, targets is list of dicts

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.cfg.model.lr, momentum=0.9, weight_decay=0.0005)
