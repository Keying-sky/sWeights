import numpy as np
from scipy import stats
from typing import Tuple

class Generator:
    def __init__(self, true_params):
        """ Initialise with true parameters."""
        self.params = true_params
        self.x_min, self.x_max = 0, 5
        self.y_min, self.y_max = 0, 10
    

    def generate_sample(self, n_events: int):
        """ Generate n_events from the mixture model."""
        # determine number of signal and background events
        n_signal = np.random.binomial(n_events, self.params[4])  # parameter: f
        n_background = n_events - n_signal
        
        # generate signal events
        x_signal = self.generate_gs(n_signal)
        y_signal = self.generate_hs(n_signal)
        
        # generate background events
        x_background = np.random.uniform(self.x_min, self.x_max, n_background)
        y_background = stats.truncnorm.rvs( (self.y_min - self.params[6]) / self.params[7],   # mu_b, sigma_b
                                           (self.y_max - self.params[6]) / self.params[7],
                                           loc=self.params[6], scale=self.params[7], size=n_background)
        
        # combine s and b, then mess up
        x = np.concatenate([x_signal, x_background])
        y = np.concatenate([y_signal, y_background])
        disorder_idx = np.random.permutation(n_events)
        
        return x[disorder_idx], y[disorder_idx]
    
    def generate_gs(self, n: int):
        """ Generate samples from Crystal Ball distribution."""
        samples = stats.crystalball.rvs(
            self.params[2],  # beta
            self.params[3],  # m
            loc=self.params[0],  # mu
            scale=self.params[1],  # sigma
            size=n
        )
        # reject samples outside [0, 5]
        mask = (samples >= self.x_min) & (samples <= self.x_max)
        samples = samples[mask]
        while len(samples) < n:
            extra = stats.crystalball.rvs(
                self.params[2], self.params[3],
                loc=self.params[0], scale=self.params[1],
                size=n - len(samples)
            )
            mask = (extra >= self.x_min) & (extra <= self.x_max)
            samples = np.concatenate([samples, extra[mask]])
        return samples[:n]
    
    def generate_hs(self, n: int):
        """ Generate samples from truncated exponential."""
        samples = np.random.exponential(1/self.params[5], n)  # lamda
        mask = (samples >= self.y_min) & (samples <= self.y_max)
        samples = samples[mask]
        while len(samples) < n:
            extra = np.random.exponential(1/self.params[5], n - len(samples))
            mask = (extra >= self.y_min) & (extra <= self.y_max)
            samples = np.concatenate([samples, extra[mask]])
        return samples[:n]
