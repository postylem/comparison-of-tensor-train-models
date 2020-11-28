import torch
import torch.nn as nn
import torch.optim as optim


class TTrain(nn.Module):
    def __init__(self, d, D, dtype, verbose=False):
        super().__init__()
        self.D = D
        self.d = d
        self.verbose = verbose
        # the following are set to nn.Parameters thus are backpropped over
        self.core = nn.Parameter(torch.rand(d, D, D, dtype=dtype))
        self.left_boundary = nn.Parameter(torch.rand(D, dtype=dtype))
        self.right_boundary = nn.Parameter(torch.rand(D, dtype=dtype))

    def _probability(self, x):
        """Unnormalized probability of one configuration P(x)
        Parameters
        ----------
        x : numpy array, shape (seqlen,)
            One configuration
        Returns
        -------
        probability : float
        """
        pass

    def _contract_at(self, x):
        """Contract network at particular values in the physical dimension,
        for computing probability of x.
        """
        # repeat the core seqlen times
        w = self.core[None].repeat(self.seqlen, 1, 1, 1)
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
        if self.verbose:
            print("contract_at", output)
        return output

    def _contract_all(self):
        """Contract network with a copy of itself across physical index,
        for computing norm.
        """
        # repeat the core seqlen times
        w = self.core[None].repeat(self.seqlen, 1, 1, 1)

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
            contracting_tensor, self.right_boundary, self.right_boundary)
        if self.verbose:
            print("contract_all", output)
        return output


    def fit(self, X, d):
        """Fit the network to the d-categorical data X
        Parameters
        ----------
        X : tensor shape (n_datapoints, seqlen)
        d : physical dimension (range of x_i)

        Returns
        -------
        self : TN
            The fitted model.
        """

        self.n_datapoints = X.shape[0]
        self.seqlen = X.shape[1]
        self.d = d

        self.norm = self._contract_all()

        # TODO: training here ...
        # self.training()

        # just for now, calculate the probability of the first datapoint
        self.probability0 = self._probability(X[0])

        return self

    def training():
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        for epoch in range(100):
            for x, target in data:
                # clear out gradients
                model.zero_grad()

                # TODO: run forward pass.
                # log_probs = logprobs(x)

                # compute the loss, gradients
                loss = loss_function(log_probs, target)
                loss.backward()
                # update the parameters
                optimizer.step()

        print('Finished Training')
