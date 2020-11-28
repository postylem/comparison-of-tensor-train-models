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
        self.D = D
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        
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

        #n_features : the number of tensor cores, i.e. length of input
        #d : physical dimension, i.e. dimension of input
        #D : bond dimension

        ###CONTRACTING THE NETWORK USING TORCH.DOT###
        #take the parameter vector and reshape it as a n_fxdxDxD, then square it
        weights_tensor = torch.tensor.reshape(self.w, (self.n_features, self.d, self.D, self.D), requires_grad=True)
        #They square the entries of the weights tensor for their calculations of the derivative later on
        #not sure if we need to square the weights if used autograd in pytorch. If not, simply replace weights_squared with weights_tensor
        #the first tensor in the network is a vector
        weights_squared = torch.square(weights_tensor[0, x[0], 0, :])
        #now contract the network, from left to right to get your probability: perform matrix vector multiplication at each step, contracting the virtual indices 
        for i in xrange(1, self.n_features-1):
            weights_squared = torch.dot(weights_squared, torch.square(weights_tensor[i, x[i], :, :])) #MPS contraction  
        #take the inner product between the built up vector (from previous contraction steps) and the end vector
        probability = torch.dot(weights_squared, torch.square(weights_tensor[self.n_features-1, x[self.n_features-1], :, 0]))

        return probability

        ###CONTRACTING RECURRENT NETWORK USING EINSUM####
        #take one core tensor of shape (d,D,D) and copy it n_features time, where n_features is the length of the sequence
        weights_tensor = self.core[None].repeat(n_features, 0)
        #contract the intiial bond dimension using left boundary vector and square the weights in order to ensure positive parameters
        contracting_tensor = torch.square(torch.einsum('i, ij -> j', self.left_boundary, weights_tensor[0, x[0], :, :]))
        #contract the network from left to right, performing a series of matrix-vector multiplications
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
        weights_tensor = torch.tensor.reshape(self.w, (self.n_features, self.d, self.D, self.D), requires_grad=True)
        weights_squared = torch.sum(torch.square(weights_tensor[0, :, 0, :]), dim=0) #First tensor
        for i in xrange(1, self.n_features-1):
            weights_squared = torch.dot(weights_squared, torch.sum(torch.square(weights_tensor[i, :, :, :]), dim=0)) #MPS contraction  
        norm = torch.dot(weights_squared, torch.sum(torch.square(weights_tensor[self.n_features-1, :, :, 0]), 0))

        return norm

        ###CONTRACTING RECURRENT NETWORK USING EINSUM####
        weights_tensor = self.core[None].repeat(n_features, 0)
        #same process as probability but must sum over physical bonds for norm
        #first tensor of network, shape(D)
        contracting_tensor = torch.sum(torch.square(torch.einsum('j, ijk -> ik', self.left_boundary, weights_tensor[0, :, :, :])), dim=0)
        #contract the network, repeated matrix vector multiplication : (D*D)@(D*DxD*D)
        for i in xrange(1, self.n_features):
            contracting_tensor = torch.einsum('i, ij -> j', contracting_tensor, torch.sum(weights_tensor[i, :, :, :], dim=0))
        norm = torch.einsum('i, i -> ', contracting_tensor, self.right_boundary)

        return norm
        
    def _derivative(self, x):
        """Compute the derivative of P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """
        w2 = np.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative = np.zeros((self.n_features,self.d,self.D,self.D))

        #Store intermediate tensor contractions for the derivatives: 
        #left to right and right to left
        #tmp stores the contraction of the first i+1 tensors from the left 
        #in tmp[i,:,:], tmp2 the remaining tensors on the right
        #the mps contracted is the remaining contraction tmp[i-1]w[i]tmp2[i+1]
        tmp = np.zeros((self.n_features,self.D))
        tmp2 = np.zeros((self.n_features,self.D))
        tmp[0,:] = np.square(w2[0,x[0],0,:])
        for i in xrange(1,self.n_features-1):
            tmp[i,:] = np.dot(tmp[i-1,:],np.square(w2[i,x[i],:,:]))  
        tmp[self.n_features-1,:] = np.inner(tmp[self.n_features-2,:],
                np.square(w2[self.n_features-1,x[self.n_features-1],:,0]))
        tmp2[self.n_features-1,:] = np.square(w2[self.n_features-1,
                x[self.n_features-1],:,0])
        for i in xrange(self.n_features-2,-1,-1):
            tmp2[i,:] = np.dot(np.square(w2[i,x[i],:]),tmp2[i+1,:])
        tmp2[0,:] = np.inner(np.square(w2[0,x[0],0,:]),tmp2[1,:])
    
        #The derivative of each tensor is the contraction of the other tensors
        derivative[0,x[0],0,:] = np.multiply(tmp2[1,:],2*(w2[0,x[0],0,:]))
        derivative[self.n_features-1,x[self.n_features-1],:,0] = \
                    np.multiply(tmp[self.n_features-2,:],
                        2*(w2[self.n_features-1,x[self.n_features-1],:,0]))
        for i in xrange(1,self.n_features-1):
                derivative[i,x[i],:,:]=np.multiply(np.outer(tmp[i-1,:],
                tmp2[i+1,:]),2*(w2[i,x[i],:]))

        return derivative.reshape(self.m_parameters)

    def _derivativenorm(self):
        """Compute the derivative of the norm
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """
        w2 = np.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative = np.zeros((self.n_features,self.d,self.D,self.D)) 
        
        tmp=np.zeros((self.n_features,self.D))
        tmp2=np.zeros((self.n_features,self.D))
        tmp[0,:]=np.sum(np.square(w2[0,:,0,:]),0)
        for i in xrange(1,self.n_features-1):
            tmp[i,:]=np.dot(tmp[i-1,:],np.sum(np.square(w2[i,:,:,:]),0)) 
        tmp[self.n_features-1,:]=np.inner(tmp[self.n_features-2,:],
                np.sum(np.square(w2[self.n_features-1,:,:,0]),0))
        tmp2[self.n_features-1,:]=np.sum(np.square(w2[self.n_features-1,:,:,0]),0)
        for i in xrange(self.n_features-2,-1,-1):
            tmp2[i,:]=np.dot(np.sum(np.square(w2[i,:,:,:]),0),tmp2[i+1,:])
        tmp2[0,:]=np.inner(np.sum(np.square(w2[0,:,0,:]),0),tmp2[1,:])
    
        for j in xrange(self.d):
            derivative[0,j,0,:]=np.multiply(tmp2[1,:],2*(w2[0,j,0,:]))
            derivative[self.n_features-1,j,:,0]=\
                np.multiply(tmp[self.n_features-2,:],2*(w2[self.n_features-1,j,:,0]))
        for i in xrange(1,self.n_features-1):
            temp3=np.outer(tmp[i-1,:],tmp2[i+1,:])
            for j in xrange(self.d):
                derivative[i,j,:,:]=np.multiply(temp3,2*(w2[i,j,:,:]))
        return derivative.reshape(self.m_parameters)

