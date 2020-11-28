# -*- coding: utf-8 -*-

from .MPSClass import TN
import numpy as np


class ComplexBorn(TN):
    """Born machine with complex parameters
    Probability is the absolute value squared of the MPS
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

        ###CONTRACTING RECURRENT NETWORK WITH EINSUM###
        #the procedure is exact same as the previous models but we take the absolute value square at the end (as opposed to just squaring)
        weights_tensor = self.core[None].repeat(n_features, 0)
        #perform left boundary contraction
        contracting_tensor = torch.einsum('i, ij -> j', self.left_boundary, weights_tensor[0, x[0], :, :])
        #contract the network
        for i in range(1, n_features):
            contracting_tensor = torch.einsum('i, ij -> j', contracting_tensor, weights_tensor[i, x[i], :, :])
        #contract last bond dimension with right boundary vector
        output = torch.einsum('i,i -> ', contracting_tensor, self.right_boundary)
        #take the absolute value to get the modulus and square it due to the born rule
        probability = torch.abs(output)**2

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
        norm = torch.einsum('ij, i, j ->', contracting_tensor, self.right_boundary, self.right_boundary)

        return norm