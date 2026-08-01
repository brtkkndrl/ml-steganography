import lightning as L
from pathlib import Path
from data import download_dataset, ImageDataModule
from model import SteganographyModel, HidingNetwork, RecoveryNetwork, DiscriminatorNetwork

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


    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto"
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    args = parser.parse_args()

    train(args.path)