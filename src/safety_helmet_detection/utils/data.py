import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def convert_xml_to_yolo(cfg: DictConfig):
    """Convert Pascal VOC XML annotations to YOLO TXT format."""
    data_dir = Path(cfg.data.data_dir)
    anns_dir = data_dir / "annotations"
    labels_dir = data_dir / "labels"

    if not anns_dir.exists():
        logger.warning(f"Annotations directory {anns_dir} not found. Skipping conversion.")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Map class name to index based on cfg.data.names (case-insensitive)
    class_map = {name.lower(): i for i, name in enumerate(cfg.data.names)}

    logger.info(f"Converting XML annotations to YOLO format in {labels_dir}...")

    for xml_file in anns_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            continue
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w == 0 or h == 0:
            continue

        yolo_data = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower()
            if cls_name not in class_map:
                continue

            cls_id = class_map[cls_name]
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )

            # YOLO format: cls x_center y_center width height (normalized)
            x_center = (b[0] + b[1]) / (2.0 * w)
            y_center = (b[2] + b[3]) / (2.0 * h)
            width = (b[1] - b[0]) / w
            height = (b[3] - b[2]) / h

            yolo_data.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_data:
            txt_path = labels_dir / f"{xml_file.stem}.txt"
            txt_path.write_text("\n".join(yolo_data))


def create_dataset_yaml(cfg: DictConfig, output_path: Path) -> Path:
    """Create YOLO-format dataset.yaml file."""
    data_dir = Path(cfg.data.data_dir).resolve()

    yaml_content = f"""path: {data_dir}
train: images
val: images

names:
"""
    for idx, name in enumerate(cfg.data.names):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = output_path / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def ensure_dataset_exists(cfg: DictConfig):
    """Check if dataset exists and download if necessary."""
    from ..data.downloader import download_data

    data_dir = cfg.data.data_dir
    data_path = Path(data_dir)

    # Download if specifically requested OR if data is missing/empty
    should_download = cfg.data.get("download", False) or not data_path.exists() or not any(data_path.iterdir())

    if should_download:
        logger.info(f"Dataset not found or empty at {data_dir}. Starting download...")
        download_data(data_dir, cfg.data.get("gdrive_folder_url"))
