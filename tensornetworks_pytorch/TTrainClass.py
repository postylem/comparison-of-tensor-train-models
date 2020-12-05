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
        if homogeneous:
            self.core = nn.Parameter(torch.rand(d, D, D, dtype=dtype))
        else:
            print("nonhomogeneous init")
            self.core = nn.Parameter(torch.rand(self.seqlen, d, D, D, dtype=dtype))
        self.left_boundary = nn.Parameter(torch.rand(D, dtype=dtype))
        self.right_boundary = nn.Parameter(torch.rand(D, dtype=dtype))

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
