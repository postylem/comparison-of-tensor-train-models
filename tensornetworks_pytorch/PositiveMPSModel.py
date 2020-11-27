# -*- coding: utf-8 -*-

from .MPSClass import TN
import numpy as np
from sklearn.externals.six.moves import xrange

class PositiveMPS(TN):
    """Matrix Product States with non-negative parameters
    Parametrization using the square of real parameters.
    Parameters
    ----------
    D : int, optional
        Rank/Bond dimension of the MPS
    learning_rate : float, optional
        Learning rate of the gradient descent algorithm
    batch_size : int, optional
        Number of examples per minibatch.
    n_iter : int, optional
        Number of iterations (epochs) over the training dataset to perform
        during training.
    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.
    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.
    ----------
    Attributes
    ----------
    w : numpy array, shape (m_parameters)
        Parameters of the tensor network
    norm : float
        normalization constant for the probability distribution
    n_samples : int
        number of training samples
    n_features : int
        number of features in the dataset
    d : int
        physical dimension (dimension of the features)
    m_parameters : int
        number of parameters in the network
    history : list
        saves the training accuracies during training
    """
    def __init__(self, D=4, learning_rate=0.1, batch_size=10,
                 n_iter=100, random_state=None, verbose=False):
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

        #n_features : the number of tensor cores, i.e. length of input P(X_1, ..., X_N) would have n_features = N
        #d : physical dimension, i.e. the possible values each x_i can take
        #D : bond dimension

        #the way this is set up now, we are taking one data point at a time, i.e. one configuration. 
        #We would need another loop if we wanted to the whole dataset as an argument: X would be an array of shape (n_samples, n_features)

        ###CONTRACTING RECURRENT NETWORK USING EINSUM####
        #take one core tensor of shape (d,D,D) and copy it n_features time, where n_features is the length of the sequence
        weights_tensor = self.core[None].repeat(n_features, 0)
        #contract the intiial bond dimension using left boundary vector and square the weights in order to ensure positive parameters
        #we treat the paramters of the different models (PMPS, BR, BC) uniformly and restrict the parameters in this case to be positive
        contracting_tensor = torch.square(torch.einsum('i, ij -> j', self.left_boundary, weights_tensor[0, x[0], :, :]))
        #contract the network from left to right, performing a series of matrix-vector multiplications
        #you are always contracting the left hand bond dimension, D
        for i in xrange(1, n_features):
            torch.einsum('i, ij -> j', contracting_tensor, torch.square(weights_tensor[i, x[i], :, :]))
        #contract the final bond dimension using the right boundary vector
        probability = torch.einsum('i, i ->', contracting_tensor, self.right_boundary)

        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """

        ###CONTRACTING RECURRENT NETWORK USING EINSUM####
        weights_tensor = self.core[None].repeat(n_features, 0)
        #same process as probability but must sum over physical bonds for norm
        #we have two copies of the same network stuck together, attached at the physical index
        #first tensor of network, shape(D)
        contracting_tensor = torch.sum(torch.square(torch.einsum('j, ijk -> ik', self.left_boundary, weights_tensor[0, :, :, :])), dim=0)
        #contract the network, repeated matrix vector multiplication : (D*D)@(D*DxD*D)
        #now, at each step, you are contracting the left bond dimension, D, and the physical dimension, d, because this connects the two copies
        for i in xrange(1, self.n_features):
            contracting_tensor = torch.einsum('i, ij -> j', contracting_tensor, torch.sum(weights_tensor[i, :, :, :], dim=0))
        #contract the final bond dimension using the right boundary vector
        norm = torch.einsum('i, i -> ', contracting_tensor, self.right_boundary)

        return norm