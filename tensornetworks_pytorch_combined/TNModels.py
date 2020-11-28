from .TNClass import TN
import torch


class PositiveMPS(TN):
    def __init__(self, d, D, verbose=False):
        super().__init__(d, D, verbose=verbose)

    def _probability(self, x):
        core = self.core.square()  # square to ensure positive parameters
        probability = self._contract_at(x, core)
        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        weights_tensor = self.core[None].repeat_interleave(
            self.n_features, dim=0)
        weights_tensor.square()  # square to ensure positive parameters

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


class RealBorn(TN):
    def __init__(self, d, D, verbose=False):
        super().__init__(d, D, verbose=verbose)

    def _probability(self, x):
        core = self.core
        output = self._contract_at(x, core)
        probability = output.square()
        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """

        weights_tensor = self.core[None].repeat(n_features, 0)
        #perform left boundary contraction
        contracting_tensor = torch.einsum(
            'ij, ik -> jk', 
            torch.einsum('j, ijk -> ik', self.left_boundary, weights_tensor[0, :, :, :]),
            torch.einsum('j, ijk -> ik', self.left_boundary, weights_tensor[0, :, :, :]))
        #contract the network
        for i in range(1, n_features):
            contracting_tensor = torch.einsum(
                'ij, ijkl -> kl',
                contracting_tensor, 
                np.einsum(
                    'ijk, ilm -> jlkm',
                    weights_tensor[i, :, :, :],
                    weights_tensor[i, :, :, :]))
        #contract the final bond dimension with right boundary vector
        norm = torch.einsum(
            'ij, i, j ->', 
            contracting_tensor, self.right_boundary, self.right_boundary)

        return norm


class ComplexBorn(TN):
    def __init__(self, d, D, verbose=False):
        super().__init__(d, D, verbose=verbose)

    def _probability(self, x):
        core = self.core
        output = self._contract_at(x, core)
        probability = output.abs().square()
        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """

        ###CONTRACTING RECURRENT NETWORK WITH EINSUM###
        #this is also the same as the other cases but for one copy of the network, we take the complex conjugate
        #take one core tensor of shape (d,D,D) and copy it n_features time, where n_features is the length of the sequence
        weights_tensor = self.core[None].repeat(n_features, 0)

        #perform left boundary contraction to 
        contracting_tensor = torch.einsum(
            'ij, ik -> jk',
            torch.einsum('j, ijk -> ik', self.left_boundary, weights_tensor[0, :, :, :]),
            torch.einsum('j, ijk -> ik', self.left_boundary, torch.conj(weights_tensor[0, :, :, :])))
        #contract the network
        for i in range(1, n_features):
            contracting_tensor = torch.einsum('ij, ijkl -> kl',
                contracting_tensor,
                np.einsum('ijk, ilm -> jlkm', weights_tensor[i, :, :, :], torch.conj(weights_tensor[i, :, :, :])))
        norm = torch.einsum(
            'ij, i, j ->', 
            contracting_tensor, self.right_boundary, self.right_boundary)

        return norm
