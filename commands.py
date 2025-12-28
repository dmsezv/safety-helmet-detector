import logging
import sys
from pathlib import Path

import fire
import hydra
import pytorch_lightning as pl
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Ensure src is in path
sys.path.append(str(Path(__file__).parent / "src"))

from safety_helmet_detection.data.datamodule import SafetyHelmetDataModule  # noqa: E402
from safety_helmet_detection.models.module import SafetyHelmetDetector  # noqa: E402

# from safety_helmet_detection.data.downloader import download_data # Imported inside module or here

logger = logging.getLogger(__name__)


class Commands:
    def _compose_config(self, overrides=None):
        if overrides is None:
            overrides = []

        GlobalHydra.instance().clear()
        # Initialize hydra
        hydra.initialize(version_base=None, config_path="configs")
        cfg = hydra.compose(config_name="config", overrides=overrides)
        return cfg

    def train(self, **kwargs):
        """
        Train the model.
        Usage: python commands.py train --parameter=value
        Example: python commands.py train --train.epochs=20
        """
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = self._compose_config(overrides)

        print("Config:\n", OmegaConf.to_yaml(cfg))
        pl.seed_everything(cfg.seed)

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

        trainer = pl.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            logger=logger_pl if logger_pl else True,
            log_every_n_steps=10,
            precision=cfg.train.get("precision", 32),
        )

        trainer.fit(model, datamodule=dm)

    def infer(self, checkpoint_path, image_path, **kwargs):
        """
        Run inference.
        """
        # Logic for inference
        print(f"Inference not fully implemented in this stub. Model loaded from {checkpoint_path}")
        # model = SafetyHelmetDetector.load_from_checkpoint(checkpoint_path)
        # model.eval()
        pass

    def download_data(self, data_dir="safety-helmet-ds"):
        from safety_helmet_detection.data.downloader import download_data

        download_data(data_dir)


if __name__ == "__main__":
    fire.Fire(Commands)
