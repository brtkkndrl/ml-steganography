import lightning as L
from pathlib import Path
from data import download_dataset, ImageDataModule
from model import SteganographyModel, HidingNetwork, RecoveryNetwork, DiscriminatorNetwork
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def train(path):
    dataset_path = download_dataset()
    dm = ImageDataModule(path=Path(dataset_path) / "imagenet", batch_size=16,
                         train_limit=4000, val_limit=1000)
    if path is None:
        model = SteganographyModel(
            hiding_network=HidingNetwork(),
            recovery_network=RecoveryNetwork(),
            discriminator_network=DiscriminatorNetwork()
        )
    else:
        model = SteganographyModel.load_from_checkpoint(
            path,
            hiding_network=HidingNetwork(),
            recovery_network=RecoveryNetwork(),
            discriminator_network=DiscriminatorNetwork()
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.4f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        min_delta=0.001
    )

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    args = parser.parse_args()

    train(args.path)