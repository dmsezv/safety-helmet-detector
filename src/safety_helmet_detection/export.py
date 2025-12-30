"""Module for exporting models to ONNX format."""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def export_model(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    model_type: str = "fasterrcnn",
    img_size: Optional[int] = None,
    opset_version: int = 12,
    **kwargs,
):
    """
    Export a trained model to ONNX format.

    Args:
        checkpoint_path: Path to the model checkpoint (.ckpt for FasterRCNN, .pt for YOLO).
        output_path: Path to save the ONNX model. If None, derived from checkpoint_path.
        model_type: Type of model ('fasterrcnn' or 'yolo').
        img_size: Image size for dummy input. If None, tries to get from config.
        opset_version: ONNX opset version.
        **kwargs: Additional arguments for export.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_path is None:
        output_path = str(checkpoint_path.with_suffix(".onnx"))

    logger.info(f"Exporting {model_type} model from {checkpoint_path} to {output_path}...")

    if model_type == "yolo":
        return _export_yolo(checkpoint_path, imgsz=img_size, **kwargs)
    else:
        return _export_fasterrcnn(checkpoint_path, output_path, img_size, opset_version, **kwargs)


def _export_yolo(checkpoint_path: Path, **kwargs):
    """Export YOLO model using ultralytics API."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Cannot export YOLO model.")
        return None

    model = YOLO(checkpoint_path)
    path = model.export(format="onnx", **kwargs)
    logger.info(f"YOLO model exported to {path}")

    return path


def _export_fasterrcnn(checkpoint_path: Path, output_path: str, img_size: Optional[int], opset_version: int, **kwargs):
    """Export FasterRCNN model using torch.onnx."""
    from .models.module import SafetyHelmetDetector

    try:
        model = SafetyHelmetDetector.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load FasterRCNN checkpoint: {e}")
        return None

    model.eval()
    model.to("cpu")

    if img_size is None:
        img_size = getattr(model.cfg.data, "img_size", 640)

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model.model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch_size", 2: "height", 3: "width"}},
    )

    try:
        import onnx

        onnx_model = onnx.load(output_path)
        meta = onnx_model.metadata_props.add()
        meta.key = "names"
        names = getattr(model.cfg.data, "names", ["helmet", "head", "person"])
        meta.value = str(list(names))

        onnx.save(onnx_model, output_path)
        logger.info(f"Added metadata to ONNX model: {names}")
    except ImportError:
        logger.warning("onnx library not found. Skipping metadata addition.")
    except Exception as e:
        logger.warning(f"Failed to add metadata: {e}")

    logger.info(f"FasterRCNN model exported to {output_path}")
    return output_path
