import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from data import download_dataset, ImageDataModule
from model import SteganographyModel, HidingNetwork, RecoveryNetwork
from pathlib import Path
import numpy as np
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr, multiscale_structural_similarity_index_measure as mssim

def evaluate(model, dm):
    cover, secret = next(iter(dm.test_dataloader()))

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

def evaluate_stats(model, dm):
    cover_psnr_vals, cover_msssim_vals, secret_psnr_vals = [], [], []

    with torch.no_grad():
        for cover, secret in dm.test_dataloader():
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    psnr_min = min(np.min(cover_psnr_vals), np.min(secret_psnr_vals))
    psnr_max = max(np.max(cover_psnr_vals), np.max(secret_psnr_vals))

    axes[0].boxplot(secret_psnr_vals)
    axes[0].set_title("Secret PSNR")
    axes[0].set_ylim(psnr_min, psnr_max)

    axes[1].boxplot(cover_psnr_vals)
    axes[1].set_title("Cover PSNR")
    axes[1].set_ylim(psnr_min, psnr_max)

    axes[2].boxplot(cover_msssim_vals)
    axes[2].set_title("Cover MS-SSIM")

    plt.tight_layout()
    plt.savefig("eval_boxplots.png")
    plt.close()

def export_onnx(model):
    device = next(model.hiding_network.parameters()).device

    dummy_cover = torch.randn(1, 3, 256, 256, device=device)
    dummy_secret = torch.randn(1, 1, 128, 128, device=device)

    torch.onnx.export(
        model.hiding_network,
        (dummy_cover, dummy_secret),
        "hiding_network.onnx",
        input_names=["cover", "secret"],
        output_names=["stego"],
        opset_version=17,
        dynamo=False
    )

    dummy_stego = torch.randn(1, 3, 256, 256, device=device)

    torch.onnx.export(
        model.recovery_network,
        dummy_stego,
        "recovery_network.onnx",
        input_names=["stego"],
        output_names=["recovered"],
        opset_version=17,
        dynamo=False
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    dataset_path = download_dataset()
    dm = ImageDataModule(path=Path(dataset_path) / "images", batch_size=16,
                         train_size=20_000, val_size=2000, test_size=2000)
    
    model = SteganographyModel.load_from_checkpoint(
        args.path,
        hiding_network=HidingNetwork(),
        recovery_network=RecoveryNetwork()
    )
    model.eval()

    dm.setup()

    evaluate_stats(model, dm)
    evaluate(model, dm)
    export_onnx(model)