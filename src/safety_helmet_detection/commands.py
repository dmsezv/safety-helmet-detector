"""Main entry point for the project CLI."""

import logging
import sys

import fire
from hydra import compose, initialize

from .export import export_model
from .infer import infer as run_infer
from .train import train as run_train

logger = logging.getLogger(__name__)


def _get_config(kwargs):
    """
    Setup Hydra and compose config.
    Captures dot-notation overrides from both kwargs and sys.argv.
    """

    # Flatten nested dicts from fire
    def flatten(d, prefix=""):
        items = []
        for k, v in d.items():
            new_key = f"{prefix}{k}"
            if isinstance(v, dict):
                items.extend(flatten(v, prefix=f"{new_key}."))
            else:
                items.append(f"{new_key}={v}")
        return items

    overrides = flatten(kwargs)

    # Add raw dot-notation overrides directly from sys.argv
    cli_overrides = [a for a in sys.argv if "=" in a and not a.startswith("-")]
    for o in cli_overrides:
        if o not in overrides:
            overrides.append(o)

    if overrides:
        logger.info(f"Overrides applied: {overrides}")

    with initialize(config_path="../../configs", version_base="1.1"):
        return compose(config_name="config", overrides=overrides)


def train(**kwargs):
    """Train the model with Hydra configuration."""
    cfg = _get_config(kwargs)
    run_train(cfg)


def export(checkpoint_path, model_type="fasterrcnn", output_path=None, **kwargs):
    """Export trained model to ONNX."""
    export_model(checkpoint_path, model_type=model_type, output_path=output_path, **kwargs)


def infer_from_checkpoint(checkpoint_path, image_path, **kwargs):
    """Run inference using a model checkpoint."""
    cfg = _get_config(kwargs)
    run_infer(cfg, checkpoint_path, image_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    fire.Fire(
        {
            "train": train,
            "export": export,
            "infer": {
                "from_checkpoint": infer_from_checkpoint,
            },
        }
    )
