import torch
import torch.nn as nn
import torch.optim as optim


class TTrain(nn.Module):
    """Abstract class for Tensor Train models.  Use instantiating class.

    Parameters:
        D (int): bond dimension
        d (int): physical dimension (number of categories in data)
        dtype ([tensor.dtype]): 
            tensor.float for real, or tensor.cfloat for complex
    """
    def __init__(self, dataset, d, D, dtype, homogeneous=True, verbose=False):
        super().__init__()
        self.D = D
        self.d = d
        self.verbose = verbose
        self.homogeneous = homogeneous
        self.n_datapoints = dataset.shape[0]
        self.seqlen = dataset.shape[1]

        # the following are set to nn.Parameters thus are backpropped over
        k_core = (d*D*D)**-0.5 
        k_vectors = (d)**-0.5
        # TODO k should be (d*D*D)**-0.5, 
        # we should use randn instead of rand, 
        # but this seems to make more NaNs for the homogeneous models.
        if homogeneous: # initialize single core to be repeated
            core = k_core * torch.randn(d, D, D, dtype=dtype)
            self.core = nn.Parameter(core)
        else: # initialize seqlen different non-homogeneous cores
            core = k_core * torch.randn(self.seqlen, d, D, D, dtype=dtype)
            self.core = nn.Parameter(core)
        self.left_boundary = nn.Parameter(k_vectors*torch.randn(D, dtype=dtype))
        self.right_boundary = nn.Parameter(k_vectors*torch.randn(D, dtype=dtype))

    def mat_norm(self, mat):
        """Our norm for matrices: infinity norm"""
        # equivalent to torch.linalg.norm(vec, ord=float('inf')).real
        return torch.max(torch.sum(abs(mat), dim=1))
        
    def vec_norm(self, vec):
        """Our norm for vectors: infinity norm"""
        # equivalent to torch.linalg.norm(vec, ord=float('inf')).real
        return vec.abs().max()

    def _log_contract_at(self, x):
        """Contract network at particular values in the physical dimension,
        for computing probability of x.
        Uses log norm stability trick.
        RETURNS A LOG PROB.
        """
        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core
        # contract the network, from the left boundary through to the last core
        Z = self.vec_norm(self.left_boundary)
        contractor_unit = self.left_boundary / Z
        accumulated_lognorm = Z.log()
        for i in range(self.seqlen):
            contractor_temp = torch.einsum(
                'i, ij -> j',
                contractor_unit,
                w[i, x[i], :, :])
            Z = self.vec_norm(contractor_temp)
            contractor_unit = contractor_temp / Z
            accumulated_lognorm += Z.log()
        # contract the final bond dimension
        output = torch.einsum(
            'i, i ->', contractor_unit, self.right_boundary)
        output = (accumulated_lognorm.exp()*output).abs().square()
        logprob = output.log()
        # if self.verbose:
        #     print("contract_at", output)
        return logprob
        
    def _contract_at(self, x):
        """Contract network at particular values in the physical dimension,
        for computing probability of x.
        """
        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core
        # contract the network, from the left boundary through to the last core
        contracting_tensor = self.left_boundary
        for i in range(self.seqlen):
            contracting_tensor = torch.einsum(
                'i, ij -> j',
                contracting_tensor,
                w[i, x[i], :, :])
        # contract the final bond dimension
        output = torch.einsum(
            'i, i ->', contracting_tensor, self.right_boundary)
        # if self.verbose:
        #     print("contract_at", output)
        return output

    def _log_contract_all(self):
        """Contract network with a copy of itself across physical index,
        for computing norm.
        """

        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core

        # first, left boundary contraction
        # (note: if real-valued conj will have no effect)
        Z = self.vec_norm(self.left_boundary)
        contractor_unit = self.left_boundary / Z
        accumulated_lognorm = Z.log()
        contractor_temp = torch.einsum(
            'ij, ik -> jk',
            torch.einsum(
                'j, ijk -> ik', contractor_unit, w[0, :, :, :]),
            torch.einsum(
                'j, ijk -> ik', contractor_unit, w[0, :, :, :].conj())
        )
        Z = self.mat_norm(contractor_temp)
        contractor_unit = contractor_temp / Z
        accumulated_lognorm += Z.log()
        # contract the network
        for i in range(1, self.seqlen):
            contractor_temp = torch.einsum(
                'ij, ijkl -> kl',
                contractor_unit,
                torch.einsum(
                    'ijk, ilm -> jlkm',
                    w[i, :, :, :],
                    w[i, :, :, :].conj()))
            Z = self.mat_norm(contractor_temp)
            contractor_unit = contractor_temp / Z
            accumulated_lognorm += Z.log()
        # contract the final bond dimension with right boundary vector
        output = torch.einsum(
            'ij, i, j ->',
            contractor_unit,
            self.right_boundary,
            self.right_boundary.conj())
        lognorm = (accumulated_lognorm.exp()*output).abs().log()
        # if self.verbose:
        #     print("contract_all", output)
        return lognorm
        

    def _contract_all(self):
        """Contract network with a copy of itself across physical index,
        for computing norm.
        """

        if self.homogeneous:
            # repeat the core seqlen times
            w = self.core[None].repeat(self.seqlen, 1, 1, 1)
        else:
            w = self.core

        # first, left boundary contraction
        # (note: if real-valued conj will have no effect)
        contracting_tensor = torch.einsum(
            'ij, ik -> jk',
            torch.einsum(
                'j, ijk -> ik', self.left_boundary, w[0, :, :, :]),
            torch.einsum(
                'j, ijk -> ik', self.left_boundary, w[0, :, :, :].conj())
        )
        # contract the network
        for i in range(1, self.seqlen):
            contracting_tensor = torch.einsum(
                'ij, ijkl -> kl',
                contracting_tensor,
                torch.einsum(
                    'ijk, ilm -> jlkm',
                    w[i, :, :, :],
                    w[i, :, :, :].conj()))
        # contract the final bond dimension with right boundary vector
        output = torch.einsum(
            'ij, i, j ->',
            contracting_tensor,
            self.right_boundary,
            self.right_boundary.conj())
        # if self.verbose:
        #     print("contract_all", output)
        return output

    def _logprob(self, x):
        """Compute log probability of one configuration P(x)

        Args:
            x (np.ndarray): shape (seqlen,)

        Returns:
            logprob (torch.Tensor): size [1]
        """
        pass

    def forward(self, x):
        return self._logprob(x)

    def train(self, data):
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        for _ in range(100):
            for x in data:
                # clear out gradients
                self.zero_grad()

                # run forward pass.
                loss =  - self(x)
                loss.backward()
                # update the parameters
                optimizer.step()

        print('Finished Training')
