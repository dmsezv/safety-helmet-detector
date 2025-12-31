import logging
import shutil
import zipfile
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)


def download_data(data_dir: str, gdrive_url: str = None):
    """Download and extract dataset from Google Drive ZIP."""
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
        gdown.download(url=gdrive_url, output=str(zip_path), quiet=False, fuzzy=True)

        if zipfile.is_zipfile(zip_path):
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path)
            zip_path.unlink()
            
            _flatten_directory(path)

        logger.info("Data preparation complete.")

    except Exception as e:
        logger.error(f"Download or preparation failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise


def _flatten_directory(path: Path):
    """Ensure images and annotations are at the root of the data_dir."""
    # Find where 'images' directory is located
    images_dirs = list(path.glob("**/images"))

    if not images_dirs:
        logger.warning(f"Could not find 'images' directory in {path}")
        return

    # Take the first 'images' directory found
    images_parent = images_dirs[0].parent

    # If it's already at the root, we are good
    if images_parent == path:
        logger.info("Dataset structure is already correct.")
        return

    logger.info(f"Correcting dataset structure: moving contents from {images_parent} to {path}")
    for item in images_parent.iterdir():
        dest = path / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))

    for item in path.iterdir():
        if item.is_dir() and item.name not in ["images", "annotations"]:
            try:
                shutil.rmtree(item)
            except Exception:
                pass
