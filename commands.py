import logging
import sys
from pathlib import Path

import fire
import hydra
from hydra.core.global_hydra import GlobalHydra

# Ensure src is in path so we can import the package if not installed
sys.path.append(str(Path(__file__).parent / "src"))

from safety_helmet_detection.data.downloader import download_data as download_task  # noqa: E402
from safety_helmet_detection.infer import infer as infer_task  # noqa: E402
from safety_helmet_detection.train import train as train_task  # noqa: E402

logger = logging.getLogger(__name__)


class Commands:
    def _compose_config(self, overrides=None):
        if overrides is None:
            overrides = []

        GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="configs")
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

    def download_data(self, data_dir="safety-helmet-ds"):
        """
        Download dataset.
        """
        download_task(data_dir)


if __name__ == "__main__":
    fire.Fire(Commands)
