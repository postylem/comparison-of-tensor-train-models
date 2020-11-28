from .TTrainClass import TTrain
import torch
import torch.nn as nn


class PositiveMPS(TTrain):
    def __init__(self, d, D, verbose=False):
        super().__init__(d, D, torch.float, verbose=verbose)

    def _probability(self, x):
        # abs core to ensure positive parameters
        self.core = nn.Parameter(self.core.abs())
        self.left_boundary = nn.Parameter(self.left_boundary.abs())
        self.right_boundary = nn.Parameter(self.right_boundary.abs())

        probability = self._contract_at(x)
        normalization = self._contract_all()
        logprob = probability.log() - normalization.log()
        return logprob


class Born(TTrain):
    """ TODO: docstring.
    Parameters:
    dtype: torch type
         torch.cfloat or torch.float """

    def __init__(self, d, D, dtype, verbose=False):
        super().__init__(d, D, dtype, verbose=verbose)

    def _probability(self, x):
        output = self._contract_at(x)
        probability = output.abs().square()
        normalization = self._contract_all().abs()
        logprob = probability.log() - normalization.log()
        if self.verbose:
            print("probability", probability)
            print("normalization", normalization)
            print("logprob", logprob)
        return logprob
