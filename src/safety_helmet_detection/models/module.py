import pytorch_lightning as pl
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    YOLO = None
    DetectionModel = None


class SafetyHelmetDetector(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        self.model_type = cfg.model.get("type", "fasterrcnn")

        if self.model_type == "fasterrcnn":
            self._init_fasterrcnn()
        elif self.model_type == "yolo":
            self._init_yolo()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _init_fasterrcnn(self):
        # Load model
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if self.cfg.model.pretrained else None)
        # Replace head for our classes (background + classes)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)

    def _init_yolo(self):
        if YOLO is None:
            raise ImportError("Ultralytics not installed. Install with `pip install ultralytics`")

        # Load a pretrained YOLO model (e.g. yolov8n.pt) or build from yaml
        # Usage context: We usually restart training or finetune.
        name = self.cfg.model.get("name", "yolov8n.pt")
        print(f"Loading YOLO model: {name}")
        self.yolo_wrapper = YOLO(name)
        # YOLO class is a wrapper, the actual torch module is .model
        # But .model might not be loaded until we call train or predict?
        # Typically strictly wrapping YOLO in PL manual loop is hard because loss is internal.
        # We will expose the internal model.
        if self.yolo_wrapper.model is None:
            # Force load
            self.yolo_wrapper._load(name)

        self.model = self.yolo_wrapper.model

        # Adjust head if needed? YOLOv8 auto-adjusts on first run usually or we can rely on finetuning.
        # But for MLOps structure, we trust typical YOLO usage.

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.model_type == "fasterrcnn":
            return self._step_fasterrcnn(batch, batch_idx)
        elif self.model_type == "yolo":
            return self._step_yolo(batch, batch_idx)

    def _step_fasterrcnn(self, batch, batch_idx):
        images, targets = batch
        # images is list of tensors, targets is list of dicts

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v)

        return loss

    def _step_yolo(self, batch, batch_idx):
        # Implementation of YOLOv8 loss inside standard PL loop is non-trivial
        # without copying extensive code from ultralytics.
        # For this homework, we provide a placeholder or suggest using standard FasterRCNN.
        # If the user REALLY wants YOLO, they typically use `yolo train ...`.
        # Here we raise error to guide user to use FasterRCNN for the PL Requirement
        # Or we can try to return a dummy loss if just testing pipeline structure.
        raise NotImplementedError(
            "YOLOv8 training inside custom PyTorch Lightning loop requires complex loss migration. "
            "Please use 'model=fasterrcnn' for the graded assignment part requiring PL training loop. "
            "Or use native `yolo` CLI for YOLO training."
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.cfg.model.lr, momentum=0.9, weight_decay=0.0005)
