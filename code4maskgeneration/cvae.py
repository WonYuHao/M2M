import torch
import torch.nn as nn
import torch.nn.functional as F


class CondVAE(nn.Module):
    def __init__(self, latent_dim=64, cond_channels=1, dropout_p=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_channels = cond_channels
        self.dropout_p = float(dropout_p)
        in_enc = 1 + cond_channels

        def d2():
            return nn.Dropout2d(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_enc, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            d2(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            d2(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            d2(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            d2(),
            nn.Flatten(),
        )

        flat_dim = 256 * 32 * 32
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        cc = cond_channels
        self.dec_up1 = nn.Sequential(nn.ConvTranspose2d(256 + cc, 128, 4, 2, 1), nn.ReLU(), d2())
        self.dec_up2 = nn.Sequential(nn.ConvTranspose2d(128 + cc, 64, 4, 2, 1), nn.ReLU(), d2())
        self.dec_up3 = nn.Sequential(nn.ConvTranspose2d(64 + cc, 32, 4, 2, 1), nn.ReLU(), d2())
        self.dec_up4 = nn.Sequential(nn.ConvTranspose2d(32 + cc, 16, 4, 2, 1), nn.ReLU(), d2())
        self.final_conv = nn.Sequential(
            nn.Conv2d(16 + cc, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = self.fc_dec(z).view(-1, 256, 32, 32)

        def cond_at(size):
            return F.interpolate(c, size=(size, size), mode="nearest")

        h = torch.cat([h, cond_at(32)], dim=1)
        h = self.dec_up1(h)
        h = torch.cat([h, cond_at(64)], dim=1)
        h = self.dec_up2(h)
        h = torch.cat([h, cond_at(128)], dim=1)
        h = self.dec_up3(h)
        h = torch.cat([h, cond_at(256)], dim=1)
        h = self.dec_up4(h)
        h = torch.cat([h, cond_at(512)], dim=1)
        return self.final_conv(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar
