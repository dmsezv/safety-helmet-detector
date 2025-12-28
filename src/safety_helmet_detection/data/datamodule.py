import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .dataset import SafetyHelmetDataset, collate_fn
from .downloader import download_data


class SafetyHelmetDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = cfg.data.data_dir

    def prepare_data(self):
        if self.cfg.data.download:
            download_data(self.data_dir)

    def setup(self, stage=None):
        # Augmentations
        train_transform = A.Compose(
            [A.Resize(self.cfg.data.img_size, self.cfg.data.img_size), A.HorizontalFlip(p=0.5), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        val_transform = A.Compose(
            [A.Resize(self.cfg.data.img_size, self.cfg.data.img_size), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        # Instantiate datasets
        ds_full = SafetyHelmetDataset(self.data_dir, transform=None)
        full_size = len(ds_full)
        train_size = int(0.8 * full_size)

        indices = torch.randperm(full_size, generator=torch.Generator().manual_seed(self.cfg.seed)).tolist()
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        # Create separate datasets with transforms
        self.train_ds = SafetyHelmetDataset(self.data_dir, transform=train_transform)
        self.val_ds = SafetyHelmetDataset(self.data_dir, transform=val_transform)

        self.train_ds = torch.utils.data.Subset(self.train_ds, train_idx)
        self.val_ds = torch.utils.data.Subset(self.val_ds, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
