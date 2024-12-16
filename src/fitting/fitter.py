import numpy as np
from scipy import stats
from iminuit import Minuit
from typing import Optional, Dict


class Fitter:
    """ Extended Maximum Likelihood Fitter."""
    def __init__(self, x, y):
        """
        Initialise the fitter.
        
        Parameters:
            x: x values array
            y: y values array
        """
        self.x = x
        self.y = y
        self.n_events = len(x)
        self.x_min, self.x_max = 0, 5
        self.y_min, self.y_max = 0, 10
        
    def crystal_norm(self, beta, m, mu, sigma):
        """ normalisation for truncated Crystal Ball. """
        return (stats.crystalball.cdf(self.x_max, beta, m, loc=mu, scale=sigma) - stats.crystalball.cdf(self.x_min, beta, m, loc=mu, scale=sigma))

    def exp_norm(self, lambda_) :
        """normalisation for truncated exponential"""
        return 1 - np.exp(-lambda_ * self.y_max)
    
    def nor_norm(self, mu, sigma):
        """normalisation for truncated normal"""
        return (stats.norm.cdf(self.y_max, mu, sigma) - stats.norm.cdf(self.y_min, mu, sigma))
    
    def neg_lnL(self, mu, sigma, beta, m, f, lamda, mu_b, sigma_b, N):                     
        """
        Calculate negative log likelihood for extended ML fit in 2-dimension.
        
        Parameters:
            mu, sigma, beta, m: Crystal Ball parameters for signal in X
            f: signal fraction
            lamda: exponential parameter for signal in Y
            mu_b, sigma_b: normal parameters for background in Y
            N: expected total number of events
        Return:
            the negative log likelihood funtion including a extended poisson term
        """
        try:
            # g_s
            cb_nor = self.crystal_norm(beta, m, mu, sigma)
            signal_x = stats.crystalball.pdf(self.x, beta, m, loc=mu, scale=sigma) / cb_nor
            
            # h_s
            exp_nor = self.exp_norm(lamda)
            signal_y = lamda * np.exp(-lamda * self.y) / exp_nor
            
            # g_b
            background_x = np.ones_like(self.x) / (self.x_max - self.x_min)
            
            # h_b
            norm_nor = self.nor_norm(mu_b, sigma_b)
            background_y = stats.norm.pdf(self.y, mu_b, sigma_b) / norm_nor
            
            # total pdf
            signal = signal_x * signal_y
            background = background_x * background_y
            total_pdf = f * signal + (1 - f) * background
            
            # numerical stability protection
            total_pdf = np.maximum(total_pdf, 1e-300)
            
            poisson_term = N - self.n_events * np.log(N)
            pdf_term = -np.sum(np.log(total_pdf))
            
            return poisson_term + pdf_term
            
        except Exception as e:
            # return a large number if computation fails
            return 1e300

    def fit(self, init_guess: Optional[Dict] = None):
        """
        Perform 2-dimensional extended maximum likelihood fit.
        
        Parameters:
            init_guess: optional dictionary of initial parameter values
        Returns:
            estimated params, parabolic errors (uncertaintis, approximately equal to the standard deviation)
        """
        if init_guess is None:
            init_guess = {'mu':3.0,'sigma':0.3,'beta':1.0,'m':1.4,'f':0.6,'lamda':0.3,'mu_b':0.0,'sigma_b':2.5,'N':float(self.n_events)}
        
        # create Minuit instance
        m = Minuit(self.neg_lnL, **init_guess)

        m.migrad() 
        m.hesse() 

        params = np.array([m.values[p] for p in m.parameters])
        errors = np.array([m.errors[p] for p in m.parameters])
        
        return params, errors

        