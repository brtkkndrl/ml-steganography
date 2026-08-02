import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from data import download_dataset, ImageDataModule
from model import SteganographyModel, HidingNetwork, RecoveryNetwork, DiscriminatorNetwork
from pathlib import Path
import numpy as np
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr, multiscale_structural_similarity_index_measure as mssim

def evaluate(path):
    dataset_path = download_dataset()
    dm = ImageDataModule(path=Path(dataset_path) / "imagenet", batch_size=16,
                              train_limit=2000, val_limit=1000)
    
    model = SteganographyModel.load_from_checkpoint(
        path,
        hiding_network=HidingNetwork(),
        recovery_network=RecoveryNetwork(),
        discriminator_network=DiscriminatorNetwork()
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

    titles = ["Cover", "Secret", "Container", "Recovered", "Residual"]
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

def evaluate_stats(path):
    dataset_path = download_dataset()
    dm = ImageDataModule(path=Path(dataset_path) / "imagenet", batch_size=16,
                          train_limit=2000, val_limit=1000)

    model = SteganographyModel.load_from_checkpoint(
        path,
        hiding_network=HidingNetwork(),
        recovery_network=RecoveryNetwork(),
        discriminator_network=DiscriminatorNetwork()
    )
    model.eval()

    dm.setup()

    cover_psnr_vals, cover_msssim_vals, secret_psnr_vals = [], [], []

    with torch.no_grad():
        for cover, secret in dm.val_dataloader():
            cover = cover.to(model.device)
            secret = secret.to(model.device)

            container, recovered = model.hide((cover, secret))

            # per-sample metrics, since batch-level averages would hide min/max
            for i in range(cover.size(0)):
                c_psnr = psnr(container[i:i+1], cover[i:i+1], data_range=1.0)
                c_msssim = mssim(container[i:i+1], cover[i:i+1], data_range=1.0)
                s_psnr = psnr(recovered[i:i+1], secret[i:i+1], data_range=1.0)

                cover_psnr_vals.append(c_psnr.item())
                cover_msssim_vals.append(c_msssim.item())
                secret_psnr_vals.append(s_psnr.item())

    def report(name, values):
        values = np.array(values)
        print(f"{name}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

    report("Cover PSNR", cover_psnr_vals)
    report("Cover MS-SSIM", cover_msssim_vals)
    report("Secret PSNR", secret_psnr_vals)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # evaluate(args.path)
    evaluate_stats(args.path)