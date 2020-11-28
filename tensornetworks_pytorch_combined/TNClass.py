import torch
import torch.nn as nn
import torch.optim as optim


class TN(nn.Module):
    def __init__(self, d, D, verbose=False):
        super().__init__()
        self.D = D
        self.d = d
        self.verbose = verbose
        # the following are set to nn.Parameters thus are backpropped over
        self.core = nn.Parameter(torch.rand(d, D, D))
        self.left_boundary = nn.Parameter(torch.rand(D))
        self.right_boundary = nn.Parameter(torch.rand(D))

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
        pass

    def _contract_at(self, x, core):
        """Contract network at particular values in the physical dimension
        """
        # repeat the core n_features times
        weights_tensor = core[None].repeat_interleave(self.n_features, dim=0)
        # contract the network, from the left boundary through to the last core
        contracting_tensor = self.left_boundary
        for i in range(self.n_features):
            contracting_tensor = torch.einsum(
                'i, ij -> j',
                contracting_tensor,
                weights_tensor[i, x[i], :, :])
        # contract the final bond dimension
        output = torch.einsum(
            'i, i ->', contracting_tensor, self.right_boundary)
        return output

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        pass

    def fit(self, X, d):
        """Fit the network to the d-categorical data X
        Parameters
        ----------
        X : tensor shape (n_samples, n_features)
        d : physical dimension (range of x_i)

        Returns
        -------
        self : TN
            The fitted model.
        """

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.d = d

        self.norm = self._computenorm()

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
