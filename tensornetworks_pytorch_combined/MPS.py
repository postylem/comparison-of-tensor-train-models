# -*- coding: utf-8 -*-

from .MPSClass import TN
import numpy as np


class PosMPS(TN):
    def __init__(self, d=10, D=4, verbose=False):
        super().__init__()

    def _probability(self, x):
        """Unnormalized probability of one configuration P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        probability : float
        """
        weights_tensor = self.core[None].repeat_interleave(n_features, dim=0)
        contracting_tensor = torch.square(
            torch.einsum(
                'i, ij -> j',
                self.left_boundary, weights_tensor[0, x[0], :, :]))
        for i in range(1, n_features):
            torch.einsum(
                'i, ij -> j',
                contracting_tensor,
                torch.square(weights_tensor[i, x[i], :, :]))
        probability = torch.einsum(
            'i, i ->', contracting_tensor, self.right_boundary)

        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """

        weights_tensor = self.core[None].repeat_interleave(n_features, dim=0)
        # same process as probability but sum over physical bonds for norm
        # we have two copies of the same network stuck together,
        # attached at the physical index
        # first tensor of network, shape(D)
        contracting_tensor = torch.einsum(
            'j, ijk -> ik',
            self.left_boundary,
            weights_tensor[0, :, :, :]).square().sum(dim=0)
        # contract: repeated matrix vector multiplication : (D*D)@(D*DxD*D)
        # now, at each step, you are contracting the left bond dimension, D,
        # and the physical dimension, d, because this connects the two copies
        for i in range(1, self.n_features):
            contracting_tensor = torch.einsum(
                'i, ij -> j',
                contracting_tensor,
                weights_tensor[i, :, :, :].sum(dim=0))
        # contract the final bond dimension using the right boundary vector
        norm = torch.einsum(
            'i, i -> ', contracting_tensor, self.right_boundary)

        return norm
