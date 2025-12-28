import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_data(data_dir: str):
    """
    Downloads data if it doesn't exist.
    This is a stub/mock as per instructions.
    In a real scenario, use dvc.api or s3 fetch.
    """
    path = Path(data_dir)
    if path.exists():
        logger.info(f"Data directory {path} already exists. Skipping download.")
        return

    logger.info(f"Downloading data to {path} (MOCK)...")
    path.mkdir(parents=True, exist_ok=True)
    # Simulate download
    (path / "readme.txt").write_text("Downloaded data placeholder")
    logger.info("Data downloaded successfully (Mock).")
