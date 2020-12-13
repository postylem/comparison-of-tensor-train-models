import numpy as np


class HMM():

    def __init__(self, n_states=8, categorical_dim=8, verbose=True):        
        self.K = n_states
        self.d = categorical_dim
        self.verbose = verbose
        self.pi = None
        self.A = None
        self.mu = None

    def parameter_initialization(self, X):
        """
        Init parameters from the data.
        this can be overloaded to do some desired initialization.
        """
        self.pi = np.array([1./ self.K] * self.K)

        self.A = np.random.RandomState(3).uniform(size=(self.K, self.K)) # rows = current state, cols = new state probas
        self.A /= A.sum(axis=1, keepdims=True)

        self.mu = np.ones((self.d, self.K, X.shape[1])) #tried this to have an extra dimension to access the observation in log_emission
        self.mu /= np.sum(self.mu, axis=1, keepdims=True)


    def log_emission(self, X) :
	    """
	    (Log) probabilities under a categorical model for a time-homogeneous HMM 

	    Inputs:
	        X: [TxN] vector of observations
	        mu: [dxKxN] matrix of latent-conditional emmision probabilities
	         
	    Returns:
	        log_eps: [TxK] matrix of log emission probabilities: log p(x_t | z_t = k)
	    """
	    T, N = X.shape
	    eps = np.ones((T, self.K))
	    for t in range(T):
	      for n in range(N):
	        for k in range(self.K):
	          #this was kind of just playing around with things, I'm not exactly sure why this works exactly
	          #but I wanted the first index in mu to access the observation because I think this just IS the probability
	          eps[t, k] *= self.mu[X[t, n], k, n] 
	    log_eps = np.log(eps)

	    return log_eps

	def log_alpha_recursion(self, X) :
	    """
	    (Log) alpha recursion for a time-homogeneous HMM with Gaussian emissions

	    Inputs:
	        X: [TxN] matrix of observations
	        A: [KxK] transition matrix
	        log_eps: [TxK] matrix of log emission probabilities: log p(x_t | z_t = k)
	        pi: [Kx1] initial latent state distribution
	    
	    Returns:
	        log_alpha: [TxK] vector containing log p(z_t , x_{1:t})
	    """
	    log_eps = self.log_emission(X)
	    T = log_eps.shape[0]
	    log_pi = np.log(self.pi) # log pmf of initial state distribution
	    log_A = np.log(self.A) # log transition probs
	    #the initial log_alpha = log[p(x1|z1=k)p(z1=k)] = log_eps[0] + log_pi[0]
	    #for each t, o_t(x_t) = p(x_t|z_t) = log_eps[t]
	    log_alpha = []
	    for t in range(T):
	      if t == 0:
	        log_alpha.append(log_eps[0] + log_pi[0])
	        continue
	      # We'd like to just do the direct way, but numerical instability:
	      # log_Aa_prev = np.log(A @ np.exp(log_alpha[t-1]))
	      # So, instead, using logsumexp:
	      log_Aa_prev = np.array([logsumexp(log_A[k]+log_alpha[t-1]) for k in range(self.K)])
	      log_alpha.append(log_eps[t] + log_Aa_prev)
	    log_alpha = np.array(log_alpha)

	    return log_alpha

	def log_beta_recursion(self, X) :
	    """
	    (Log) beta recursion for a time-homogeneous HMM with Gaussian emissions

	    Inputs:
	        X: [TxN] matrix of observations
	        A: [KxK] transition matrix
	        log_eps: [TxK] matrix of log emission probabilities: log p(x_t | z_t = k)
	    
	    Returns:
	        log_beta: [TxK] vector containing log p(x_{t+1:T} | z_t)
	    """
	    log_eps = self.log_emission(X)
	    T = log_eps.shape[0]
	    log_A = np.log(self.A) # log transition probs
	    log_beta = np.zeros((T,self.K))
	    for t in reversed(range(T)):
	      if t == T-1:
	        continue
	      log_beta[t] = np.array([logsumexp(
	          log_A[:,k]+log_eps[t+1]+log_beta[t+1]
	          ) for k in range(self.K)])

	    # This redundant line is just a reminder to leave the T-th row (python's (T-1)-th)
	    # empty: Look at the equation for beta(z_T). We do this to keep equal sizes
	    # for alpha and beta recursions.
	    log_beta[T-1] = np.zeros((self.K,))

	    return log_beta

	def smoothing(self, X):
	    """
	    Smoothing probabilities for a time-homogeneous HMM with Gaussian emissions

	    Inputs:
	        log_alpha: [TxK] matrix containing log p(z_t , x_{1:t})
	        log_beta: [TxK] matrix containing log p(z_{t+1:T} | z_t)
	    
	    Returns:
	        gamma: [TxK] matrix of smoothing probabilities p(z_t | x_{1:T})
	    """
	    log_alpha = self.log_alpha_recursion(X)
	    log_beta = self.log_beta_recursion(X)
	    T = log_alpha.shape[0]

	    # We can calculate the log likelihood Z with any value of t, 
	    # they'll all be equal, so I could just use one value of t, 
	    # but this is the same, subtracting each log_Z at each t.
	    log_Z = logsumexp(log_alpha + log_beta, axis=1).reshape(T,1)
	    log_gamma = log_alpha + log_beta - log_Z
	    gamma = np.exp(log_gamma)

	    return gamma

	def pair_marginals(self, X):
        """
	    Pair marginals for a time-homogeneous HMM with Gaussian emissions

	    Inputs:
	        log_alpha: [TxK] matrix containing log p(z_t , x_{1:t})
	        log_beta: [TxK] matrix containing log p(z_{t+1:T} | z_t)
	        A: [KxK] transition matrix
	        log_eps: [TxK] matrix of log emission probabilities: log p(x_t | z_t = k)
	    
	    Returns:
	        psi: [TxKxK] numpy tensor of pair marginal probabilities p(z_t, z_{t+1} | x_{1:T})
	    """
	    log_eps = self.log_emission(X)
	    log_alpha = self.log_alpha_recursion(X)
	    log_beta = self.log_beta_recursion(X)
	    T = log_alpha.shape[0]
	    log_Z = logsumexp(log_alpha + log_beta, axis=1).reshape(T,1,1) # log likelihood
	    # These two need to be evaluated at t+1, so shift them along axis 0
	    log_beta_eps_next = np.roll(log_beta + log_eps, -1, axis=0)
	    log_psi = -log_Z \
	              + log_alpha[:,np.newaxis,:] \
	              + np.log(self.A[np.newaxis,:,:]) \
	              + log_beta_eps_next[:,:,np.newaxis]
	    #put the indices in the order we want them, with z_t first and z_t+1 second
	    log_psi = log_psi.swapaxes(1,2) 
	    
	    # What this does is  
	    #log_Z = logsumexp(log_alpha + log_beta, axis=1)[0]
	    #log_psi = np.zeros((T,K,K))
	    #for t in range(T-1):
	    #  for k1 in range(K):
	    #    for k2 in range(K):
	    #      log_psi[t,k1,k2] = -log_Z + \
	    #      (log_alpha[t,k1] + log_beta[t+1,k2]+ log_eps[t+1,k2]+ np.log(A[k2,k1]))

	    psi = np.exp(log_psi)

	    # Just as above, we keep psi of length T on the first dimension
	    psi[T-1] = np.zeros((self.K,self.K))

	    return psi

	def E_step(self, X):
	    """
	    Gets E-step estimates and log likelihood
		for data X, given parameters pi, mu, A
	    """

	    T, N = X.shape
	    #print(np.sum(self.A, axis=1))

	    #TODO
	    log_eps = self.log_emission(X)
	    log_alpha = self.log_alpha_recursion(X)
	    log_beta = self.log_beta_recursion(X)
	    gamma = self.smoothing(X)
	    #print(gamma)
	    psi = self.pair_marginals(X)
	    llike = logsumexp(log_alpha[-1], axis=0) 

	    return gamma, psi, llike/X.shape[0]

	def M_step(self, X, gamma, psi):
	    """
	    Find updated values for parameters given data
		  
	    Inputs:
	    X: [TxN] matrix of training observations
	    gamma: [TxK] p(z_t=k | x_{1:T})
	    psi: [TxKxK] p(z_t=k, z_{t+1}=j | x_{1:T})
	    d: dimension of the categorical output
	    """

	    pi_ = gamma[0] / np.sum(gamma[0])
	    self.pi = pi_

	    A_ = np.sum(psi, axis=0) 
	    A_ /= np.sum(A, axis=1, keepdims=True)
	    self.A = A_

	    mu_ = np.empty((d, self.K, X.shape[1]))
	    for d_ in range(self.d):
	      for n in range(X.shape[1]):
	        x_ = X[:, n]
	        ind = (x_ == d_)[:, np.newaxis]
	        mu_[d_, :, n] = np.sum((ind * gamma), axis=0) / np.sum(gamma, axis=0)
	    self.mu = mu_

	def exp_max(self, X):
	    """
	    Estimates the parameters of an HMM using training data X via the EM algorithm

	    Inputs:
	        X_tr: [T_trainx2] matrix of training observations
	        X_tr: [T_testx2] matrix of test observations
	        init_params: tuple of initialization parameters from previous question
	        
	    Returns:
	        pi: [K] estimated initial latent distribution
	        A: [KxK] estimated transition matrix
	        mus: [Kx2] matrix of estimated emission means
	        sigmas: [Kx2x2] tensor of estimated emission covariance matrices
	        train_avg_llike: list containing the average training log likelihood on each iteration
	        test_avg_llike: list containing the average test log likelihood on each iteration
	    """

	    # Set initialization from parameters given in previous question 
	    self.parameter_initialization(X)

	    # Train EM
	    train_avg_llike  = []
	    #test_avg_llike  = []
	    # hyperparams
	    hyp = dict(init_llike = -1e6, stopping = 1e-6, s_lim = 20)
	    llike, llike_prev, step = hyp["init_llike"], -1e2, 0
	    while (np.abs(llike - llike_prev) > hyp["stopping"]):
	      if step > hyp["s_lim"]:
	        print("Step limit reached.")
	        break
	      step += 1
	      llike_prev = llike
	      #E Step update
	      gamma, psi, llike = self.E_step(X)
	      train_avg_llike.append(llike)
	      # Get testset's llike, for plotting validation
	      #_, _, llike_ts = E_step(Xs, pi, p, A)
	      #test_avg_llike.append(llike_ts)
	      #print(step, "\ntrain llike ", llike)
	      # print(step, "\ntrain llike ", llike, "\ntest llike  ", llike_ts, sep='')
	      # print("difference in train llike", np.abs(llike - llike_prev))
	      #M Step update
	      self.M_step(X, gamma, psi)
	      #print(np.sum(A, axis=1))
	      
	    return train_avg_llike










