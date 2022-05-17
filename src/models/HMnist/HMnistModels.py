import torch

from src.models.Mnist.MnistVIB import MnistVIB
from src.utils.model_utils import get_gp_prior


class HMnistVIB(MnistVIB):

    def step(self, batch):
        x_full, x_miss, x_mask, y = batch
        x = x_miss.reshape(x_miss.shape[0], -1)
        r_z, qy_z = self.forward(x)

        log_likelihood = self.compute_log_likelihood(qy_z, y)

        kl = self.compute_kl_divergence(r_z)
        kl = torch.where(torch.torch.isfinite(kl), kl, torch.zeros_like(kl))

        elbo = log_likelihood - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo
        return loss, kl.mean(), (-log_likelihood).mean(), torch.argmax(qy_z.mean, dim=1), qy_z.mean


class HMnistGPVIB(HMnistVIB):

    def __init__(self, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, time_length=10, **kwargs):
        super().__init__(**kwargs)

    def _get_prior(self):
        if self.prior is None:
            self.prior = get_gp_prior(kernel=self.hparams.kernel, kernel_scales=self.hparams.kernel_scales,
                                      time_length=self.hparams.time_length, sigma=self.hparams.sigma,
                                      length_scale=self.hparams.length_scale, z_dim=self.hparams.z_dim,
                                      device=self.device)
        return self.prior

    def encode(self, x):
        return self.encoder(x.permute(0, 2, 1))

    def decode(self, z):
        return self.decoder(z)

    def step(self, batch):
        x_full, x, x_mask, y = batch
        r_z, qy_z = self.forward(x)

        log_likelihood = self.compute_log_likelihood(qy_z, y)

        kl = self.compute_kl_divergence(r_z)
        kl = torch.where(torch.torch.isfinite(kl), kl, torch.zeros_like(kl))
        kl = kl.sum(-1)

        elbo = log_likelihood - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo
        return loss, kl.mean(), (-log_likelihood).mean(), torch.argmax(qy_z.mean, dim=1), qy_z.mean
