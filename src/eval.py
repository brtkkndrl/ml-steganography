import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from data import download_dataset, ImageDataModule
from model import SteganographyModel, HidingNetwork, RecoveryNetwork
from pathlib import Path

def evaluate(path):
    dataset_path = download_dataset()
    dm = ImageDataModule(path=Path(dataset_path) / "imagenet", batch_size=16,
                              train_limit=2000, val_limit=1000)
    
    model = SteganographyModel.load_from_checkpoint(
        path,
        hiding_network=HidingNetwork(),
        recovery_network=RecoveryNetwork()
    )
    model.eval()

    dm.setup()
    cover, secret = next(iter(dm.train_dataloader()))

    cover = cover[:4].to(model.device)
    secret = secret[:4].to(model.device)

    with torch.no_grad():
        container, recovered = model.hide((cover, secret))

    residual = (cover - container).abs()
    residual = (residual - residual.amin(dim=(1,2,3), keepdim=True)) / \
           (residual.amax(dim=(1,2,3), keepdim=True) - residual.amin(dim=(1,2,3), keepdim=True) + 1e-8)

    titles = ["Cover", "Secret", "Container", "Recovered", "Residual x10"]
    images = [cover, secret, container, recovered, residual]

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    for col, (title, batch) in enumerate(zip(titles, images)):
        for row in range(4):
            img = batch[row].detach().cpu().permute(1, 2, 0).clamp(0, 1)
            if img.shape[-1] == 1:
                axes[row, col].imshow(img.squeeze(-1), cmap="gray")
            else:
                axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.path)