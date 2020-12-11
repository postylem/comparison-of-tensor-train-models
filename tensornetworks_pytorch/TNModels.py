from .TTrainClass import TTrain
import torch
import torch.nn as nn


class PosMPS(TTrain):
    """MPS model for tensor network with positive parameters.

    Uses absolute value of real parameters.
    """

    def __init__(
            self, dataset, d, D, 
            homogeneous=True, w_randomization=None, log_stability=True, verbose=False):
        super().__init__(
            dataset, d, D, dtype=torch.float, 
            homogeneous=homogeneous, w_randomization=w_randomization, verbose=verbose)
        self.log_stability = log_stability
        self.name = "Positive MPS"
        if homogeneous:
            self.name += ", Homogeneous"
        else:
            self.name += ", Non-homogeneous"
        if log_stability:
            self.name += " + log_stability"


    def _logprob(self, x):
        """Compute log probability of one configuration P(x)

        Args:
            x (np.ndarray): shape (seqlen,)

        Returns:
            logprob (torch.Tensor): size []
        """
        if self.log_stability:
            unnorm_logprob = self._log_contract_at(x)
            log_normalization = self._log_contract_all()
            logprob = unnorm_logprob - log_normalization
            # print(output, unnorm_prob, normalization, logprob)
        else:
            unnorm_prob = self._contract_at(x)
            normalization = self._contract_all()
            logprob = unnorm_prob.log() - normalization.log()
        return logprob


    def _contract_at(self, x):
        """Contract network at particular values in the physical dimension,
        for computing probability of x.
        """
        # repeat the core seqlen times
        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core
        w2 = w.square()
        left_boundary2 = self.left_boundary.square()
        right_boundary2 = self.right_boundary.square()
        # contract the network, from the left boundary through to the last core
        contracting_tensor = left_boundary2
        for i in range(self.seqlen):
            contracting_tensor = torch.einsum(
                'i, ij -> j',
                contracting_tensor,
                w2[i, x[i], :, :])
            if contracting_tensor.min() < 0:
                print("contraction < 0")
                print(w.min())
        # contract the final bond dimension
        output = torch.einsum(
            'i, i ->', contracting_tensor, right_boundary2)
        # if self.verbose:
        #     print("contract_at", output)
        if output < 0:
            print("output of contract_at < 0")
        return output

    def _contract_all(self):
        """Contract network with a copy of itself across physical index,
        for computing norm.
        """
        # repeat the core seqlen times
        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core
        w2 = w.square()
        left_boundary2 = self.left_boundary.square()
        right_boundary2 = self.right_boundary.square()
        # first, left boundary contraction
        # (note: if real-valued conj will have no effect)
        contracting_tensor = torch.einsum(
            'j, ijk -> k', left_boundary2, w2[0, :, :, :])
        # contract the network
        for i in range(1, self.seqlen):
            contracting_tensor = torch.einsum(
                'j, ijk -> k',
                contracting_tensor,
                w2[i, :, :, :])
        # contract the final bond dimension with right boundary vector
        output = torch.dot(contracting_tensor, right_boundary2)
        # if self.verbose:
        #     print("contract_all", output)
        return output

    def _log_contract_at(self, x):
        """Contract network at particular values in the physical dimension,
        for computing probability of x.
        """
        # repeat the core seqlen times
        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core
        w2 = w.square()
        left_boundary2 = self.left_boundary.square()
        right_boundary2 = self.right_boundary.square()
        Z = self.vec_norm(left_boundary2)
        contractor_unit = left_boundary2 / Z
        accumulated_lognorm = Z.log()
        # contract the network, from the left boundary through to the last core
        #contracting_tensor = left_boundary2
        for i in range(self.seqlen):
            contractor_temp = torch.einsum(
                'i, ij -> j',
                contractor_unit,
                w2[i, x[i], :, :])
            Z = self.vec_norm(contractor_temp)
            contractor_unit = contractor_temp / Z
            accumulated_lognorm += Z.log()
            if contractor_unit.min() < 0:
                print("contraction < 0")
                print(w.min())
        # contract the final bond dimension
        output = torch.einsum(
            'i, i ->', contractor_unit, right_boundary2)
        logprob = accumulated_lognorm + output.log()
        # if self.verbose:
        #     print("contract_at", output)
        if output < 0:
            print("output of contract_at < 0")
        return logprob

    def _log_contract_all(self):
        """Contract network with a copy of itself across physical index,
        for computing norm.
        """
        # repeat the core seqlen times
        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core
        w2 = w.square()
        left_boundary2 = self.left_boundary.square()
        right_boundary2 = self.right_boundary.square()
        Z = self.vec_norm(left_boundary2)
        contractor_unit = left_boundary2 / Z
        accumulated_lognorm = Z.log()
        # first, left boundary contraction
        # (note: if real-valued conj will have no effect)
        contractor_temp = torch.einsum(
            'j, ijk -> k', contractor_unit, w2[0, :, :, :])
        Z = self.vec_norm(contractor_temp)
        contractor_unit = contractor_temp / Z
        accumulated_lognorm += Z.log()
        # contract the network
        for i in range(1, self.seqlen):
            contractor_temp = torch.einsum(
                'j, ijk -> k',
                contractor_unit,
                w2[i, :, :, :])
            Z = self.vec_norm(contractor_temp)
            contractor_unit = contractor_temp / Z
            accumulated_lognorm += Z.log()
        # contract the final bond dimension with right boundary vector
        output = torch.dot(contractor_unit, right_boundary2)
        logprob = accumulated_lognorm + output.log()
        # if self.verbose:
        #     print("contract_all", output)
        return logprob

    # def _contract_at_temp(self, x):
    #     """Contract network at particular values in the physical dimension,
    #     for computing probability of x.
    #     """
    #     # repeat the core seqlen times
    #     w2 = self.core[None].repeat(self.seqlen, 1, 1, 1).square()
    #     left_boundary2 = self.left_boundary.square()
    #     right_boundary2 = self.right_boundary.square()
    #     contracting_tensor = left_boundary2
    #     for i in range(self.seqlen):
    #         contracting_tensor = torch.einsum(
    #             'i, ij -> j',
    #             contracting_tensor,
    #             w2[i, x[i], :, :]) 
    #     output = torch.dot(contracting_tensor, right_boundary2)
    #     return output

    # def _contract_all_temp(self):
    #     """Contract network with a copy of itself across physical index,
    #     for computing norm.
    #     """
    #     w2 = self.core[None].repeat(self.seqlen, 1, 1, 1).square()
    #     left_boundary2 = self.left_boundary.square()
    #     right_boundary2 = self.right_boundary.square()
    #     contracting_tensor = torch.einsum('j, ijk -> ik', left_boundary2, w2[0,:,:,:])
    #     contracting_tensor = contracting_tensor.sum(axis=0)
    #     for i in range(1, self.seqlen):
    #         contracting_tensor = torch.einsum(
    #             'i, ij -> j', 
    #             contracting_tensor, 
    #             torch.sum(w2[i,:,:,:], axis=0))  
    #     output = torch.dot(contracting_tensor, right_boundary2)
    #     return output


class Born(TTrain):
    """Born model for tensor network with real or complex parameters.

    Parameters:
        dtype ([tensor.dtype]): 
            tensor.float for real, or tensor.cfloat for complex
    """
    def __init__(
            self, dataset, d, D, dtype, 
            homogeneous=True, w_randomization=None, log_stability=True, verbose=False):
        super().__init__(
            dataset, d, D, dtype, 
            homogeneous, w_randomization=w_randomization, verbose=verbose)
        self.log_stability = log_stability
        self.name = f"Born ({dtype})"
        if homogeneous:
            self.name += ", Homogeneous"
        else:
            self.name += ", Non-homogeneous"
        if log_stability:
            self.name += " + log_stability"

    def _logprob(self, x):
        """Compute log probability of one configuration P(x)

        Args:
            x (np.ndarray): shape (seqlen,)

        Returns:
            logprob (torch.Tensor): size []
        """
        if self.log_stability:
            unnorm_logprob = self._log_contract_at(x)
            log_normalization = self._log_contract_all()
            logprob = unnorm_logprob - log_normalization
            # print(output, unnorm_prob, normalization, logprob)
        else:
            output = self._contract_at(x)
            unnorm_prob = output.abs().square()
            normalization = self._contract_all().abs()
            logprob = unnorm_prob.log() - normalization.log()
        return logprob

    def _logprob_batch(self, X):
        """Compute log P(x) for all x in a batch X

        Args:
            X : shape (batch_size, seqlen)

        Returns:
            logprobs (torch.Tensor): size [batchsize]
        """
        if self.log_stability:
            unnorm_logprobs = self._log_contract_at_batch(X) # tensor size [batchsize]
            # print(unnorm_logprobs)
            # print([self._log_contract_at(x).item() for x in X])
            log_normalization = self._log_contract_all() # scalar
            logprobs = unnorm_logprobs - log_normalization
        else:
            raise NotImplementedError('batched=True not implemented for log_stability=False')
        return logprobs