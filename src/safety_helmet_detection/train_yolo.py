"""YOLO training using native ultralytics API."""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def create_dataset_yaml(cfg: DictConfig, output_path: Path) -> Path:
    """Create YOLO-format dataset.yaml file."""
    data_dir = Path(cfg.data.data_dir).resolve()

    yaml_content = f"""path: {data_dir}
train: images
val: images

names:
"""
    for idx, name in enumerate(cfg.data.names):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = output_path / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def get_device(cfg: DictConfig) -> str:
    """Convert config accelerator to YOLO device string."""
    if cfg.train.accelerator == "gpu":
        return "0"
    return "cpu"


def train_yolo(cfg: DictConfig):
    """Train YOLO using ultralytics native training."""
    logger.info(f"YOLO config:\n{OmegaConf.to_yaml(cfg)}")

    import os

    from ultralytics import YOLO

    output_dir = Path("outputs/yolo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow integration for YOLO
    if cfg.logger.get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = cfg.logger.tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.logger.experiment_name
        logger.info(f"MLflow tracking enabled for YOLO: {cfg.logger.tracking_uri}")

    dataset_yaml = create_dataset_yaml(cfg, output_dir)
    model = YOLO(cfg.model.get("name", "yolov8n.pt"))

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

    # Auto-export to ONNX
    best_model_path = output_dir / "train" / "weights" / "best.pt"
    if best_model_path.exists():
        logger.info(f"Exporting best model to ONNX: {best_model_path}")
        model.export(format="onnx")

    return results
