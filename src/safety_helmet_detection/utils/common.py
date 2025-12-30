from omegaconf import DictConfig


def get_device(cfg: DictConfig) -> str:
    """Convert config accelerator to YOLO-compatible device string."""
    if cfg.train.accelerator == "gpu":
        return "0"
    if cfg.train.accelerator == "mps":
        return "mps"
    return "cpu"
