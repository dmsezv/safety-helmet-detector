"""Training orchestration module."""

import logging

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from .data.datamodule import SafetyHelmetDataModule
from .export import export_model
from .models.module import SafetyHelmetDetector

logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    """Main entry point for training. Routes to appropriate backend."""
    model_type = cfg.model.get("type", "fasterrcnn")

    if model_type == "yolo":
        from .train_yolo import train_yolo

        train_yolo(cfg)
    else:
        train_lightning(cfg)


def train_lightning(cfg: DictConfig):
    """Train model using PyTorch Lightning."""
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed)

    datamodule = SafetyHelmetDataModule(cfg)
    model = SafetyHelmetDetector(cfg)

    pl_logger = _create_logger(cfg)
    trainer = _create_trainer(cfg, pl_logger)

    trainer.fit(model, datamodule=datamodule)

    # Export best model to ONNX
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        logger.info(f"Exporting best model to ONNX: {best_path}")
        export_model(best_path, model_type="fasterrcnn")


def _create_logger(cfg: DictConfig):
    """Create MLFlow logger if tracking_uri is configured."""
    if not cfg.logger.get("tracking_uri"):
        return True

    return MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        log_model=cfg.logger.log_model,
    )


def _create_trainer(cfg: DictConfig, pl_logger):
    """Create PyTorch Lightning Trainer."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename="best_model",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    return pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=pl_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.train.get("log_every_n_steps", 10),
        precision=cfg.train.get("precision", 32),
    )
