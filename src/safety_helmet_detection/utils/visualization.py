import cv2
import numpy as np
import torch


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor = None,
    class_map: dict = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.

    Args:
        image: Image as HWC numpy array (RGB).
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        labels: Tensor of shape (N,) with class indices.
        scores: Optional tensor of shape (N,) with confidence scores.
        class_map: Mapping from class ID to class name.
        color: RGB tuple for box color.
        thickness: Line thickness.

    Returns:
        Numpy array with drawn boxes.
    """
    # Clone image to avoid modifying original
    img_draw = image.copy()

    # Reverse identity map if not provided
    id_to_name = {v: k for k, v in class_map.items()} if class_map else {}

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().tolist()
        label_id = int(labels[i])

        # Draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)

        # Text label
        label_text = id_to_name.get(label_id, str(label_id))
        if scores is not None:
            score = float(scores[i])
            label_text += f" {score:.2f}"

        # Draw text background
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(img_draw, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_draw
