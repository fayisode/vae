import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """Encoder for the VAE."""

    def __init__(self, nc: int, ndf: int, latent_dim: int):
        """Initializes the Encoder.

        Args:
            nc: Number of input channels.
            ndf: Number of encoder filters.
            latent_dim: Dimensionality of the latent space.
        """
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                self.ndf * 2, self.ndf * 4, 3 if nc == 1 else 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, 1024, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, latent_dim)
        self.fc22 = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.

        Args:
            x: Input tensor (batch_size, nc, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mu, logvar)
        """
        conv = self.encoder(x)
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)


class Decoder(nn.Module):
    """Decoder for the VAE."""

    def __init__(self, ngf: int, nc: int, latent_dim: int):
        """Initializes the Decoder.

        Args:
            ngf: Number of decoder filters.
            nc: Number of output channels.
            latent_dim: Dimensionality of the latent space.
        """
        super().__init__()
        self.ngf = ngf
        self.nc = nc
        self.latent_dim = latent_dim

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.ngf * 8, self.ngf * 4, 3 if nc == 1 else 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            z: Latent vector (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed output (batch_size, nc, height, width).
        """
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 1024, 1, 1)
        return self.decoder(decoder_input)


class MyModel(nn.Module):
    """Main VAE model."""

    def __init__(
        self, latent_dim: int, nClusters: int, nc: int = 1, ndf: int = 64, ngf: int = 64
    ):
        """Initializes the VAE model.

        Args:
            latent_dim: Dimensionality of the latent space.
            nClusters: Number of clusters in the mixture of Gaussians prior.
            nc: Number of input/output channels.
            ndf: Number of encoder filters.
            ngf: Number of decoder filters.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.nClusters = nClusters
        self.nc = nc
        self.ndf = ndf
        self.ngf = ngf

        self.encoder = Encoder(nc, ndf, latent_dim)
        self.decoder = Decoder(ngf, nc, latent_dim)

        self.pi_ = nn.Parameter(
            torch.FloatTensor(
                self.nClusters,
            ).fill_(1)
            / self.nClusters,
            requires_grad=True,
        )
        self.prior_frozen = False

        self.mu_c = nn.Parameter(
            torch.FloatTensor(self.nClusters, self.latent_dim).normal_(0, 0.01)
        )
        self.log_var_c = nn.Parameter(
            torch.FloatTensor(self.nClusters, self.latent_dim).fill_(-2.0)
        )  # log(sigma^2) ~ 0.1

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAEs.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def gaussian_pdf_log(
        self, x: torch.Tensor, mu: torch.Tensor, log_sigma2: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the Gaussian probability density function.

        Args:
            x: Input tensor.
            mu: Mean of the Gaussian.
            log_sigma2: Log variance of the Gaussian.

        Returns:
            torch.Tensor: Log probabilities.
        """
        return -0.5 * (
            torch.sum(
                torch.log(torch.tensor(np.pi * 2.0).float().to(x.device))
                + log_sigma2
                + (x - mu).pow(2) / torch.exp(log_sigma2),
                1,
            )
        )

    def gaussian_pdfs_log(
        self, x: torch.Tensor, mus: torch.Tensor, log_sigma2s: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the Gaussian probability density function for multiple clusters.

        Args:
            x: Input tensor.
            mus: Means of the Gaussians (nClusters, latent_dim).
            log_sigma2s: Log variances of the Gaussians (nClusters, latent_dim).

        Returns:
            torch.Tensor: Log probabilities (batch_size, nClusters).
        """
        G = torch.zeros(x.shape[0], self.nClusters).to(x.device)
        for c in range(self.nClusters):
            G[:, c] = self.gaussian_pdf_log(
                x, mus[c : c + 1, :], log_sigma2s[c : c + 1, :]
            ).view(-1)
        return G

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE.

        Args:
            x: Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (reconstructed_x, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = torch.nan_to_num(z, nan=0)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predicts cluster assignments for input data.

        Args:
            x: Input tensor.

        Returns:
            np.ndarray: Cluster assignments (batch_size,).
        """
        z_mu, z_sigma2_log = self.encode(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        z = torch.nan_to_num(z, nan=0)
        pi = self.pi_
        log_sigma2_c = self.log_var_c
        mu_c = self.mu_c
        yita_c = torch.exp(
            torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
        )
        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes input data to latent space.

        Args:
            x: Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mu, logvar)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector to output space.

        Args:
            z: Latent vector.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        return self.decoder(z)

    def RE(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss.

        Args:
            recon_x: Reconstructed output.
            x: Original input.

        Returns:
            torch.Tensor: Reconstruction loss.
        """
        recon_x = torch.clamp(recon_x, 0, 1)
        if x.max() > 1 or x.min() < 0:
            x = x / 255.0  # Normalize if needed
        x = torch.clamp(x, 0, 1)
        flat_size = recon_x[0].numel()  # Elements in one sample
        self.flat_size = flat_size
        return torch.nn.functional.mse_loss(
            recon_x.view(-1, flat_size),  # Flattened reconstruction
            x.view(-1, flat_size),  # Flattened target
            size_average=False,
            reduction="mean",
        )

    def KLD(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL divergence loss (Mixture of Gaussians prior).

        Args:
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        det = 1e-10
        pi = self.pi_
        log_var_c = self.log_var_c
        mu_c = self.mu_c
        z = torch.randn_like(mu) * torch.exp(log_var / 2) + mu
        z = torch.nan_to_num(z, nan=0)

        yita_c = (
            torch.exp(
                torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_var_c)
            )
            + det
        )
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))

        loss = 0.5 * torch.mean(
            torch.sum(
                yita_c
                * torch.sum(
                    log_var_c.unsqueeze(0)
                    + torch.exp(log_var.unsqueeze(1) - log_var_c.unsqueeze(0))
                    + (mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2)
                    / torch.exp(log_var_c.unsqueeze(0)),
                    2,
                ),
                1,
            )
        )
        loss -= torch.mean(
            torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)
        ) + 0.5 * torch.mean(torch.sum(1 + log_var, 1))
        return loss

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        BETA_COEF: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combined loss function.

        Args:
            recon_x: Reconstructed output.
            x: Original input.
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.
            BETA_COEF: Weight for the KL divergence loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (reconstruction_loss, kld_loss, total_loss)
        """
        reconst_loss = self.RE(recon_x, x)
        kld_loss = self.KLD(mu, log_var)
        loss = reconst_loss + kld_loss * BETA_COEF
        return reconst_loss, kld_loss, loss

    def freeze_prior(self):
        """Freeze prior parameters for initial training stability."""
        for param in [self.pi_, self.mu_c, self.log_var_c]:
            param.requires_grad = False
        self.prior_frozen = True

    def unfreeze_prior(self):
        """Unfreeze prior parameters after initial epochs."""
        for param in [self.pi_, self.mu_c, self.log_var_c]:
            param.requires_grad = True
        self.prior_frozen = False
