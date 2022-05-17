import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.model_utils import make_nn, make_cnn, make_cnn_decoder
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.multinomial import Multinomial


class Decoder(nn.Module):
    def __init__(self, hidden_sizes, num_sample=1):
        """ Decoder parent class with no specified output distribution
            :param hidden_sizes: tuple of hidden layer sizes. The tuple length sets the number of hidden layers.
        """
        super(Decoder, self).__init__()
        self.net = make_nn(hidden_sizes[-1], hidden_sizes[:-1])
        self.num_sample = num_sample

    def __call__(self, x):
        pass


class BernoulliDecoder(Decoder):
    """ Decoder with Bernoulli output distribution (used for HMNIST) """

    def __call__(self, x):
        logits = self.net(x)
        if self.num_sample > 1:
            logits = F.softmax(logits, dim=2).mean(0)
        return Bernoulli(logits=logits)


class MultinomialDecoder(Decoder):
    def __call__(self, x):
        logits = self.net(x)
        if self.num_sample > 1:
            logits = F.softmax(logits, dim=2).mean(0)
        return Multinomial(logits=logits)


class CnnMultinomialDecoder(Decoder):
    def __init__(self, hidden_sizes, kernel_size, padding, num_sample=1):
        """ Decoder parent class with no specified output distribution
            :param hidden_sizes: tuple of hidden layer sizes. The tuple length sets the number of hidden layers.
        """
        super(Decoder, self).__init__()
        self.net = make_cnn_decoder(hidden_sizes, kernel_size, padding)
        self.num_sample = num_sample

    def __call__(self, x):
        logits = self.net(x)
        if self.num_sample > 1:
            logits = F.softmax(logits, dim=2).mean(0)
        return Multinomial(logits=logits)


class GaussianDecoder(Decoder):
    """ Decoder with Gaussian output distribution (used for SPRITES and Physionet) """

    def __call__(self, x):
        mean = self.net(x)
        return Normal(loc=mean, scale=torch.ones(mean.shape, device=mean.device))
