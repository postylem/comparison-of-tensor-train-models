from .TNClass import TN
import torch
import torch.nn as nn


class PositiveMPS(TN):
    def __init__(self, d, D, verbose=False):
        super().__init__(d, D, verbose=verbose)

    def _probability(self, x):
        # square core to ensure positive parameters
        self.core = nn.Parameter(self.core.square())
        probability = self._contract_at(x)
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
        output = self._contract_at(x)
        probability = output.square()
        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """

        # n_features : the number of tensor cores, i.e. length of input P(X_1, ..., X_N) would have n_features = N
        # d : physical dimension, i.e. the possible values each x_i can take
        # D : bond dimension

        # the way this is set up now, we are taking one data point at a time, i.e. one configuration. 
        # We would need another loop if we wanted to the whole dataset as an argument: X would be an array of shape (n_samples, n_features)


        # TODO: ensure core is real-valued, but other than that,
        #       this should be the same operation as for the complex Born.
        norm = self._contract_all()

        return norm


class ComplexBorn(TN):
    def __init__(self, d, D, verbose=False):
        super().__init__(d, D, verbose=verbose)

    def _probability(self, x):
        output = self._contract_at(x)
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
        norm = self._contract_all()

        return norm
