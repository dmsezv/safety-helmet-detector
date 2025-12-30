import logging
import zipfile
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)


def download_data(data_dir: str, gdrive_url: str = None):
    """Download and extract dataset ZIP from Google Drive."""
    if not gdrive_url:
        logger.warning("No GDrive URL provided. Skipping download.")
        return

    path = Path(data_dir)
    if path.exists() and any(path.iterdir()):
        logger.info(f"Data directory {path} already exists and is not empty.")
        return

    logger.info(f"Downloading dataset from {gdrive_url}...")
    path.mkdir(parents=True, exist_ok=True)
    zip_path = path / "data.zip"

    try:
        gdown.download(url=gdrive_url, output=str(zip_path), quiet=False)

        if zipfile.is_zipfile(zip_path):
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path)
            zip_path.unlink()

        logger.info("Data preparation complete.")

    except Exception as e:
        logger.error(f"Download or extraction failed: {e}")
        raise
