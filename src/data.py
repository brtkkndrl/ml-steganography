from torch.utils.data import Dataset
from PIL import Image
import kagglehub
from pathlib import Path
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning as L
import torch

def download_dataset():
    path = kagglehub.dataset_download("lijiyu/imagenet")
    return path

class PairedImageDataset(Dataset):
    def __init__(self, path: Path, cover_transform=None, secret_transform = None):
        self.paths = sorted(Path(path).glob("*.JPEG"))
        self.cover_transform = cover_transform
        self.secret_transform = secret_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        cover_path = self.paths[idx]
        secret_idx = random.randrange(len(self.paths))
        secret_path = self.paths[secret_idx]

        cover = Image.open(cover_path).convert("RGB")
        secret = Image.open(secret_path).convert("RGB")

        if self.cover_transform:
            cover = self.cover_transform(cover)
        if self.secret_transform:
            secret = self.secret_transform(secret)

        return cover, secret

class ImageDataModule(L.LightningDataModule):
    def __init__(self, path: str, batch_size: int = 32,
                cover_size: int = 256, secret_size: int = 128, num_workers: int = 4,
                train_limit: int = None, val_limit: int = None):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_limit = train_limit
        self.val_limit = val_limit

        self.cover_transform = transforms.Compose([
            transforms.Resize(cover_size),
            transforms.RandomCrop((cover_size, cover_size)),
            transforms.ToTensor(),
        ])

        self.secret_transform = transforms.Compose([
            transforms.Resize(secret_size),
            transforms.RandomCrop((secret_size, secret_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_dataset = PairedImageDataset(self.path / "train",
                                cover_transform=self.cover_transform,
                                secret_transform=self.secret_transform)

        if self.train_limit:
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset,
                range(self.train_limit)
            )
        
        self.val_dataset = PairedImageDataset(self.path / "val",
                                cover_transform=self.cover_transform,
                                secret_transform=self.secret_transform)

        if self.val_limit:
            self.val_dataset = torch.utils.data.Subset(
                self.val_dataset,
                range(self.val_limit)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )