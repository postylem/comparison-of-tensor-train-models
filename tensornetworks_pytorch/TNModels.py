from .TTrainClass import TTrain
import torch
import torch.nn as nn


class PosMPS(TTrain):
    """MPS model for tensor network with positive parameters.

    Uses absolute value of real parameters.
    """

    def __init__(self, d, D, verbose=False):
        super().__init__(d, D, torch.float, verbose=verbose)
        self.name = "Positive MPS"

    def _logprob(self, x):
        """Compute log probability of one configuration P(x)

        Args:
            x (np.ndarray): shape (seqlen,)

        Returns:
            logprob (torch.Tensor): size []
        """
        # abs core to ensure positive parameters
        self.core = nn.Parameter(self.core.abs())
        self.left_boundary = nn.Parameter(self.left_boundary.abs())
        self.right_boundary = nn.Parameter(self.right_boundary.abs())

        unnorm_prob = self._contract_at(x)
        normalization = self._contract_all()
        logprob = unnorm_prob.log() - normalization.log()
        if self.verbose:
            # print("unnorm_prob", unnorm_prob)
            # print("normalization", normalization)
            print("logprob", logprob)
        return logprob


class Born(TTrain):
    """Born model for tensor network with real or complex parameters.

    Parameters:
        dtype ([tensor.dtype]): 
            tensor.float for real, or tensor.cfloat for complex
    """

    def __init__(self, d, D, dtype, verbose=False):
        super().__init__(d, D, dtype, verbose=verbose)
        self.name = "Born model " + repr(dtype) 

    def _logprob(self, x):
        """Compute log probability of one configuration P(x)

        Args:
            x (np.ndarray): shape (seqlen,)

        Returns:
            logprob (torch.Tensor): size []
        """
        output = self._contract_at(x)
        unnorm_prob = output.abs().square()
        normalization = self._contract_all().abs()
        logprob = unnorm_prob.log() - normalization.log()
        if self.verbose:
            # print("unnorm_prob", unnorm_prob)
            # print("normalization", normalization)
            print("logprob", logprob)
        return logprob
