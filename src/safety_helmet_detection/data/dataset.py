import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class SafetyHelmetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        # Ensure images and annotations match
        self.images_dir = self.root / "images"
        self.anns_dir = self.root / "annotations"

        self.imgs = sorted(list(self.images_dir.glob("*.png")))

        # User classes: Person, Helmet, Head
        # Mapping to ID (1-based, 0 is background)
        self.class_map = {
            "helmet": 1,
            "head": 2,
            "person": 3,
            # Case variations just in case
            "Helmet": 1,
            "Head": 2,
            "Person": 3,
        }

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # Corresponding XML
        ann_path = self.anns_dir / f"{img_path.stem}.xml"

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        boxes = []
        labels = []

        if ann_path.exists():
            tree = ET.parse(ann_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in self.class_map:
                    continue

                label = self.class_map[name]
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        if self.transform:
            # Albumentations requires bboxes
            transformed = self.transform(image=img_np, bboxes=boxes, labels=labels)
            img_tensor = transformed["image"]
            boxes_t = transformed["bboxes"]
            labels_t = transformed["labels"]
        else:
            # Fallback if no transform (should minimally be to tensor)
            # But assume caller provides transform including ToTensorV2
            img_tensor = ToTensorV2()(image=img_np)["image"]
            boxes_t = boxes
            labels_t = labels

        # Convert to torch
        if len(boxes_t) > 0:
            boxes_t = torch.tensor(boxes_t, dtype=torch.float32)
            labels_t = torch.tensor(labels_t, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_t
        target["labels"] = labels_t
        target["image_id"] = torch.tensor([idx])

        # Normalization to 0-1 happens in ToTensorV2 if float? No, ToTensorV2 implies 0-255 if uint8.
        # Usually ToFloat(max_value=255) is needed before ToTensorV2 or normalize.
        # I'll rely on the transform pipeline.

        img_tensor = img_tensor.float() / 255.0  # Simple normalization if not in transform

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))
