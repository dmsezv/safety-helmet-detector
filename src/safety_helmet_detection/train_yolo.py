"""YOLO training using native ultralytics API."""

import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from .utils import convert_xml_to_yolo, create_dataset_yaml, ensure_dataset_exists, get_device

logger = logging.getLogger(__name__)


def train_yolo(cfg: DictConfig):
    """Train YOLO using ultralytics native training."""
    logger.info(f"YOLO config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path("outputs/yolo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    ensure_dataset_exists(cfg)
    convert_xml_to_yolo(cfg)

    # MLflow integration
    if cfg.logger.get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = cfg.logger.tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.logger.experiment_name
        logger.info(f"MLflow tracking enabled: {cfg.logger.tracking_uri}")

    dataset_yaml = create_dataset_yaml(cfg, output_dir)
    model = YOLO(cfg.model.get("name", "yolov8m.pt"))

    results = model.train(
        data=str(dataset_yaml),
        epochs=cfg.train.epochs,
        imgsz=cfg.data.img_size,
        batch=cfg.data.batch_size,
        lr0=cfg.model.lr,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        device=get_device(cfg),
        workers=cfg.data.num_workers,
        momentum=cfg.model.get("momentum", 0.937),
        weight_decay=cfg.model.get("weight_decay", 0.0005),
        verbose=True,
    )

    logger.info(f"Training complete: {output_dir / 'train'}")

    best_model_path = output_dir / "train" / "weights" / "best.pt"
    if best_model_path.exists():
        logger.info(f"Exporting best model to ONNX: {best_model_path}")
        model.export(format="onnx")

    return results
