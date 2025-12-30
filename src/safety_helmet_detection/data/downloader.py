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
            # Try as a folder first if it's a folder link
            if "folders" in gdrive_url:
                logger.info("Detected Google Drive folder link.")
                gdown.download_folder(url=gdrive_url, output=str(path), quiet=False, use_cookies=False)
            else:
                logger.info("Detected Google Drive file link (e.g. ZIP).")
                zip_path = path / "data.zip"
                gdown.download(url=gdrive_url, output=str(zip_path), quiet=False)

                if zip_path.suffix == ".zip":
                    logger.info("Extracting ZIP archive...")
                    import zipfile

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(path)
                    zip_path.unlink()  # Delete zip after extraction

            logger.info("Data downloaded successfully.")
        except Exception as e:
            if "FolderContentsMaximumLimitError" in str(e):
                logger.error(
                    "\n" + "=" * 60 + "\nERROR: Google Drive folder has >50 files. gdown cannot download it directly."
                    "\nSUGGESTION: Please ZIP your dataset folder on Google Drive and use the link to the ZIP file."
                    "\n" + "=" * 60
                )
            else:
                logger.error(f"Failed to download data: {e}")
            raise e
    else:
        logger.warning("No GDrive URL provided. Skipping download.")
