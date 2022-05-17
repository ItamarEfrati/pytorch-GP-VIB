import torch
import pytorch_lightning as pl

from abc import ABC, abstractmethod


class AbstractVIB(ABC, pl.LightningModule):
    """
    VAE working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self,
                 image_preprocessor,
                 encoder,
                 decoder,
                 z_dim=256,
                 lr=1e-3,
                 weight_decay=0.005,
                 beta=1.0):
        super(AbstractVIB, self).__init__()
        self.save_hyperparameters(ignore=['image_preprocessor', 'encoder', 'decoder'])

        self.image_preprocessor = image_preprocessor
        self.encoder = encoder
        self.decoder = decoder

        self.prior = None

    @abstractmethod
    def _get_prior(self):
        pass

    @abstractmethod
    def step(self, batch):
        pass

    @abstractmethod
    def compute_log_likelihood(self, px_z, x):
        pass

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def compute_kl_divergence(self, q):
        kl = torch.distributions.kl.kl_divergence(q, self._get_prior())
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl

    # region Pytorch lightning overwrites

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def forward(self, x):
        if self.image_preprocessor:
            x = self.image_preprocessor(x)
        r_z = self.encode(x)
        z = r_z.rsample()
        qy_z = self.decode(z)
        return r_z, qy_z

    # endregion

    def get_latent_vectors(self, data_loader):
        labels = []
        latent_vectors = []
        for batch in data_loader:
            x, y = batch
            q_z = self.encoder(x)
            z = q_z.rsample()
            latent_vectors.append(z)
            labels.append(y)
        return torch.concat(latent_vectors), torch.concat(labels)

