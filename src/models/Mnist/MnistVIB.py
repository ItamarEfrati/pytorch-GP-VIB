from typing import Optional

import torch
import torchmetrics
from torch.distributions import MultivariateNormal

from src.models.abstracts.VIB import AbstractVIB


class MnistVIB(AbstractVIB):
    """
    VIB working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.train_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(num_classes=num_classes)

    def _get_prior(self):
        if self.prior is None:
            self.prior = MultivariateNormal(loc=torch.zeros(self.hparams.z_dim, device=self.device),
                                            covariance_matrix=torch.eye(self.hparams.z_dim, device=self.device))
        return self.prior

    def step(self, batch):
        x, y = batch
        r_z, qy_z = self.forward(x)

        log_likelihood = self.compute_log_likelihood(qy_z, y)

        kl = self.compute_kl_divergence(r_z)
        kl = torch.where(torch.torch.isfinite(kl), kl, torch.zeros_like(kl))

        elbo = log_likelihood - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo
        return loss, kl.mean(), (-log_likelihood).mean(), torch.argmax(qy_z.mean, dim=1), qy_z.mean

    # region Loss computations

    def compute_log_likelihood(self, qy_z, y):
        log_likelihood = qy_z.log_prob(torch.nn.functional.one_hot(y.long(), self.hparams.num_classes))
        log_likelihood = torch.where(torch.isfinite(log_likelihood), log_likelihood, torch.zeros_like(log_likelihood))
        return log_likelihood

    def compute_kl_divergence(self, r_z):
        return torch.distributions.kl.kl_divergence(r_z, self._get_prior())

    # endregion

    # region Pytorch lightning overwrites

    def training_step(self, batch, batch_idx):
        loss, kl_mean, negative_log_likelihood, predictions, probabilities = self.step(batch)

        self.train_accuracy(preds=predictions, target=batch[-1])
        self.train_auroc(probabilities, batch[-1])

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_kl_mean', kl_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mean_negative_log_likelihood', negative_log_likelihood, on_step=False, on_epoch=True,
                 prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, kl_mean, negative_log_likelihood, predictions, probabilities = self.step(batch)

        self.val_accuracy(preds=predictions, target=batch[-1])
        self.val_auroc(probabilities, batch[-1])

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_kl_mean', kl_mean, on_step=False, on_epoch=True)
        self.log('val_mean_negative_log_likelihood', negative_log_likelihood, on_step=False, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, kl_mean, negative_log_likelihood, predictions, probabilities = self.step(batch)

        self.test_accuracy(preds=predictions, target=batch[-1])
        self.test_auroc(probabilities, batch[-1])

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_kl_mean', kl_mean, on_step=False, on_epoch=True)
        self.log('test_mean_negative_log_likelihood', negative_log_likelihood, on_step=False, on_epoch=True)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # endregion
