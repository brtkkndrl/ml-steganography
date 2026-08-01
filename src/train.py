import lightning as L
from pathlib import Path
from data import download_dataset, ImageDataModule
from model import SteganographyModel, HidingNetwork, RecoveryNetwork

def train():
    dataset_path = download_dataset()
    dm = ImageDataModule(path=Path(dataset_path) / "imagenet", batch_size=16,
                         train_limit=2000, val_limit=1000)
    model = SteganographyModel(
        hiding_network=HidingNetwork(),
        recovery_network=RecoveryNetwork()
    )
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto"
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train()