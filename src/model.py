import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F

# class PrepNetwork(nn.Module):
#     def __init__(self, C_in=1, C_out=256):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv2d(C_in, 32, kernel_size=3, stride=2, padding=1), # 128 -> 64
#             nn.ReLU(),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 -> 32
#             nn.ReLU(),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),

#             nn.Conv2d(128, C_out, kernel_size=1),
#         )

#     def forward(self, x):
#         return self.net(x)

class HidingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        C_secret_prep = 64
        C_cover_prep = 64

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

            nn.Conv2d(96, 96, 5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 256 -> 128
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 3, stride=1, padding=1),  # patch-level real/fake logits
        )

    def forward(self, x):
        return self.net(x)  # returns a spatial map of logits, not a single scalar


class SteganographyModel(L.LightningModule):
    def __init__(self, hiding_network, recovery_network, discriminator_network, lr=1e-4):
        super().__init__()

        self.hiding_network = hiding_network
        self.recovery_network = recovery_network
        self.discriminator_network = discriminator_network

        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        cover, secret = batch
        opt_g, opt_d = self.optimizers()

        stego = self.hiding_network(cover, secret)
        recovered = self.recovery_network(stego)

        # Discriminator step
        real_logits = self.discriminator_network(cover)
        fake_logits = self.discriminator_network(stego.detach())
        d_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + \
                F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        d_loss = d_loss / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # Generator step
        logits_adv = self.discriminator_network(stego)
        loss_adv = F.binary_cross_entropy_with_logits(logits_adv, torch.ones_like(logits_adv))

        loss_cover = self.loss_fn(stego, cover)
        loss_secret = self.loss_fn(recovered, secret)

        beta, gamma = 1.0, 0.01
        g_loss = loss_cover + beta * loss_secret + gamma * loss_adv

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("d_loss", d_loss)
        self.log("g_loss", g_loss)
        self.log("cover_loss", loss_cover)
        self.log("secret_loss", loss_secret)
        self.log("adv_loss", loss_adv)

    def hide(self, batch):
        cover, secret = batch
        
        stego = self.hiding_network(cover, secret)
        recovered = self.recovery_network(stego)

        return stego, recovered

    def validation_step(self, batch, batch_idx):
        cover, secret = batch
        
        stego = self.hiding_network(cover, secret)
        recovered = self.recovery_network(stego)

        logits_adv = self.discriminator_network(stego)
        loss_adv = F.binary_cross_entropy_with_logits(logits_adv, torch.ones_like(logits_adv))

        loss_cover = self.loss_fn(stego, cover)
        loss_secret = self.loss_fn(recovered, secret)

        beta, gamma = 1.0, 0.01
        loss = loss_cover + beta * loss_secret + gamma * loss_adv

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
        opt_g = torch.optim.Adam(
            list(self.hiding_network.parameters()) + list(self.recovery_network.parameters()),
            lr=self.lr
        )
        opt_d = torch.optim.Adam(self.discriminator_network.parameters(), lr=self.lr)
        return [opt_g, opt_d]
