import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F

class HidingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        C_secret_prep = 64
        C_cover_prep = 128

        self.cover_prep_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 256 -> 128
            nn.ReLU(),
        )

        self.cover_prep_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 128 -> 64
            nn.ReLU(),
        )

        self.cover_prep_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),

            nn.Conv2d(128, C_cover_prep, kernel_size=1),
        )

        self.secret_prep_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 128 -> 64
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),

            nn.Conv2d(64, C_secret_prep, kernel_size=1),
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(C_secret_prep + C_cover_prep,
                                64, kernel_size=4, stride=2, padding=1), # 32 -> 64
            nn.ReLU(),
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1), # 64 -> 128 (+cover_prep_2 skip)
            nn.ReLU(),
        )

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 3, kernel_size=4, stride=2, padding=1), # 128 -> 256 (+cover_prep_1 skip)
            nn.Sigmoid(),
        )

    def forward(self, x_cover, x_secret):
        c1 = self.cover_prep_1(x_cover)   # 128x128x32
        c2 = self.cover_prep_2(c1)        # 64x64x64
        c3 = self.cover_prep_3(c2)        # 32x32xC_cover_prep

        s = self.secret_prep_net(x_secret)  # 32x32xC_secret_prep

        x = torch.cat([c3, s], dim=1)
        x = self.decoder_1(x)               # 64x64x64

        x = torch.cat([x, c2], dim=1)
        x = self.decoder_2(x)               # 128x128x32

        x = torch.cat([x, c1], dim=1)
        x = self.decoder_3(x)               # 256x256x3

        return x

class RecoveryNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(96, 96, 5, stride=2, padding=2),  # 256 -> 128
            nn.ReLU(),

            nn.Conv2d(96, 64, 3, padding=1),  # no stride, stays 128
            nn.ReLU(),

            nn.Conv2d(64, 1, 1),
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

        self.beta = 0.5   # secret reconstruction loss coefficient
        self.delta = 1.5  # L1 coefficient

    def training_step(self, batch, batch_idx):
        cover, secret = batch

        stego = self.hiding_network(cover, secret)
        recovered = self.recovery_network(stego)

        loss_cover_l2 = F.mse_loss(stego, cover)
        loss_cover_l1 = F.l1_loss(stego, cover)
        loss_cover = loss_cover_l2 + self.delta * loss_cover_l1

        loss_secret_l2 = F.mse_loss(recovered, secret)
        loss_secret_l1 = F.l1_loss(recovered, secret)
        loss_secret = loss_secret_l2 + self.delta * loss_secret_l1

        loss = loss_cover + self.beta * loss_secret

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

        loss_cover_l2 = F.mse_loss(stego, cover)
        loss_cover_l1 = F.l1_loss(stego, cover)
        loss_cover = loss_cover_l2 + self.delta * loss_cover_l1

        loss_secret_l2 = F.mse_loss(recovered, secret)
        loss_secret_l1 = F.l1_loss(recovered, secret)
        loss_secret = loss_secret_l2 + self.delta * loss_secret_l1

        loss = loss_cover + self.beta * loss_secret

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cover_loss", loss_cover, on_step=False, on_epoch=True)
        self.log("val_secret_loss", loss_secret, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.hiding_network.parameters()) + list(self.recovery_network.parameters()),
            lr=self.lr
        )