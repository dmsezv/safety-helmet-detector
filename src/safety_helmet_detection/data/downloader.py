import logging
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)


def download_data(data_dir: str, gdrive_url: str = None):
    """
    Downloads data if it doesn't exist using gdown.
    """
    path = Path(data_dir)
    if path.exists() and any(path.iterdir()):
        logger.info(f"Data directory {path} already exists and is not empty. Skipping download.")
        return

    logger.info(f"Downloading data to {path}...")
    path.mkdir(parents=True, exist_ok=True)

    if gdrive_url:
        try:
            gdown.download_folder(url=gdrive_url, output=str(path), quiet=False, use_cookies=False)
            logger.info("Data downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise e
    else:
        logger.warning("No GDrive URL provided. Skipping download.")
