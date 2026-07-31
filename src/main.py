import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image

import kagglehub

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision

import random

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
                cover_size: int = 256, secret_size: int = 256, num_workers: int = 4,
                train_limit: int = None, val_limit: int = None):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_limit = train_limit
        self.val_limit = val_limit

        self.cover_transform = transforms.Compose([
            transforms.Resize(cover_size), # TODO not sure
            transforms.RandomCrop((cover_size, cover_size)),
            transforms.ToTensor(),
        ])

        self.secret_transform = transforms.Compose([
            transforms.Resize(secret_size), # TODO not sure
            transforms.RandomCrop((secret_size, secret_size)),
            # transforms.Grayscale(num_output_channels=1),
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



import torch.nn as nn
import torch

class PrepNetwork(nn.Module):
    def __init__(self, C_in=3, C_out=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(C_in, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(64, C_out, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)

    

class HidingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        C_prep = 32

        self.prep_network = PrepNetwork(C_out=C_prep)

        self.net = nn.Sequential(
            nn.Conv2d(3+C_prep, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(64, 3, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x_cover, x_secret):
        x_prepped = self.prep_network(x_secret)
        x = torch.cat([x_cover, x_prepped], dim=1)
        return self.net(x)


class RecoveryNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SteganographyModel(L.LightningModule):
    def __init__(self, hiding_network, recovery_network, lr=1e-4):
        super().__init__()

        self.hiding_network = hiding_network
        self.recovery_network = recovery_network

        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        cover, secret = batch

        stego = self.hiding_network(cover, secret)
        recovered = self.recovery_network(stego)

        loss_cover = self.loss_fn(stego, cover)
        loss_secret = self.loss_fn(recovered, secret)

        beta = 1.0

        loss = loss_cover + beta*loss_secret

        self.log("train_loss", loss)
        self.log("cover_loss", loss_cover)
        self.log("secret_loss", loss_secret)

        return loss

    def hide(self, batch):
        cover, secret = batch
        
        stego = self.hiding_network(cover, secret)
        recovered = self.recovery_network(stego)

        return stego, recovered

    def validation_step(self, batch, batch_idx):
        cover, secret = batch

        stego = self.hiding_network(cover, secret)
        recovered = self.recovery_network(stego)

        loss_cover = self.loss_fn(stego, cover)
        loss_secret = self.loss_fn(recovered, secret)

        beta = 1.0

        loss = loss_cover + beta*loss_secret

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.log("val_cover_loss", loss_cover, on_step=False, on_epoch=True)
        self.log("val_secret_loss", loss_secret, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )
        return optimizer



if __name__ == "__main__":
    dataset_path = download_dataset()
    dm = ImageDataModule(path=Path(dataset_path) / "imagenet", batch_size=16,
                          train_limit=2000, val_limit=1000)

    model = SteganographyModel(
        hiding_network=HidingNetwork(),
        recovery_network=RecoveryNetwork()
    )

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto"
    )

    trainer.fit(model, datamodule=dm)

    # dm.setup()

    cover, secret = next(iter(dm.train_dataloader()))

    cover = cover[:4].to(model.device)
    secret = secret[:4].to(model.device)

    model.eval()
    with torch.no_grad():
        container, recovered = model.hide((cover, secret))

    titles = ["Cover", "Secret", "Container", "Recovered"]
    images = [cover, secret, container, recovered]

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for row, (title, batch) in enumerate(zip(titles, images)):
        for col in range(4):
            img = batch[col].detach().cpu().permute(1, 2, 0).clamp(0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=12)

    plt.tight_layout()
    plt.show()


    # model = SteganographyModel.load_from_checkpoint(
    #     "lightning_logs/version_5/checkpoints/epoch=4-step=625.ckpt",
    #     hiding_network=HidingNetwork(),
    #     recovery_network=RecoveryNetwork()
    # )

    # TODO stegonography model -> get container and recovered to display it

    # dm.setup()

    # cover_batch, secret_batch = next(iter(dm.train_dataloader()))

    # print(cover_batch.shape)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # cover_grid = make_grid(cover_batch, nrow=4)
    # axes[0].imshow(cover_grid.permute(1, 2, 0))
    # axes[0].set_title("Cover")
    # axes[0].axis("off")

    # secret_grid = make_grid(secret_batch, nrow=4)
    # axes[1].imshow(secret_grid.permute(1, 2, 0))
    # axes[1].set_title("Secret")
    # axes[1].axis("off")

    # plt.tight_layout()
    # plt.show()