import logging

import fire
import hydra
from hydra.core.global_hydra import GlobalHydra

from .data.downloader import download_data as download_task
from .infer import infer as infer_task
from .train import train as train_task

logger = logging.getLogger(__name__)


class Commands:
    def _compose_config(self, overrides=None):
        if overrides is None:
            overrides = []

        GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../../configs")
        cfg = hydra.compose(config_name="config", overrides=overrides)
        return cfg

    def train(self, **kwargs):
        """
        Train the model.
        Usage: python commands.py train --parameter=value
        Example: python commands.py train --train.epochs=20
        """
        # Parse overrides
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = self._compose_config(overrides)

        # Call the training pipeline
        train_task(cfg)

    def infer(self, checkpoint_path, image_path, **kwargs):
        """
        Run inference.
        """
        infer_task(checkpoint_path, image_path, **kwargs)

    def download_data(self, data_dir="safety-helmet-ds", gdrive_url=None):
        """
        Download dataset.
        """
        download_task(data_dir, gdrive_url)


if __name__ == "__main__":
    fire.Fire(Commands)
