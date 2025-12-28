import logging

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from .data.datamodule import SafetyHelmetDataModule
from .models.module import SafetyHelmetDetector

logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    """
    Main training routine.
    """
    logger.info(f"Training configuration:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed)

    # Initialize DataModule and Model
    dm = SafetyHelmetDataModule(cfg)
    model = SafetyHelmetDetector(cfg)

    # Logger setup
    logger_pl = None
    if cfg.logger.get("tracking_uri"):
        from pytorch_lightning.loggers import MLFlowLogger

        logger_pl = MLFlowLogger(
            experiment_name=cfg.logger.experiment_name,
            tracking_uri=cfg.logger.tracking_uri,
            log_model=cfg.logger.log_model,
        )

    # Init Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=logger_pl if logger_pl else True,
        log_every_n_steps=10,
        precision=cfg.train.get("precision", 32),
    )

    # Start Training
    trainer.fit(model, datamodule=dm)
