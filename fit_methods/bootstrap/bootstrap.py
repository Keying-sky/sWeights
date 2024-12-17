from dataclasses import dataclass
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib

#from src.generation.generator import Generator
from ..generation import Generator
from ..fitting import Fitter

@dataclass
class BootstrapResult:
    """Store results from bootstrap analysis."""
    sample_size: int
    lambda_hat: np.ndarray
    N_hat: np.ndarray
    lambda_mean: float
    lambda_std: float
    N_hat_mean: float
    N_hat_std: float
    lambda_bias: float

class Bootstrap:
    def __init__(self, true_params, n_ensemble=250):
        """
        Initialise the parametric bootstrapping simulation.
        
        Parameters:
            true_params: array including true parameter values
            n_ensemble: number of bootstrap iterations
        """
        self.true_params = true_params
        self.n_ensemble = n_ensemble
        self.sample_sizes = [500, 1000, 2500, 5000, 10000]
        
    def toy_study(self) -> Dict[int, BootstrapResult]:
        """Run parametric bootstrapping (toy study) for all sample sizes."""
        results = {}
        
        # iteration sample sizes
        for size in self.sample_sizes:
            lambda_fit = []
            N = []

            # 250 generations and fits per sizes
            for _ in range(self.n_ensemble):
                # add Poisson variation to sample size
                actual_size = np.random.poisson(size)
      
                x, y = Generator(self.true_params).generate_sample(actual_size)
                
                # fit with extended ML
                fitter = Fitter(x, y)
                fit_result = fitter.fit()
                
                lambda_fit.append(fit_result[0][5])  # Lambda parameter
                N.append(fit_result[0][-1])
            
            lambda_fit = np.array(lambda_fit)
            N = np.array(N)
            
            results[size] = BootstrapResult(
                sample_size=size,
                lambda_hat=lambda_fit,
                N_hat=N,
                lambda_mean=np.mean(lambda_fit),
                lambda_std=np.std(lambda_fit),
                N_hat_mean=np.mean(N),
                N_hat_std=np.std(N),
                lambda_bias=np.mean(lambda_fit) - self.true_params[5])
       
        return results
    
    def uncertainties(self, results: Dict[int, BootstrapResult]):
        """ Uncertainty in λ vs sample size."""

        sizes = np.array(list(results.keys()))
        uncertainties = np.array([r.lambda_std for r in results.values()])
        biases = np.array([r.lambda_bias for r in results.values()])

        matplotlib.rcParams['figure.dpi'] = 200 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # plot standard deviation
        ax1.errorbar(sizes, uncertainties, yerr=uncertainties/np.sqrt(2*(self.n_ensemble-1)), fmt='o-', capsize=5)       
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('SE on λ')
        ax1.set_title('Uncertainty on λ as a function of the sample size')
        
        # add data details
        for i, (x, y) in enumerate(zip(sizes, uncertainties)):
            if i == len(sizes)-1:
                ax1.annotate(f'({x}, {y:.4f})', xy=(x, y), xytext=(-80, 5), textcoords='offset points', fontsize=8)
            else:     
                ax1.annotate(f'({x}, {y:.4f})', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
                        
        # add theoritical line for comparison
        ref_point = uncertainties[0] * np.sqrt(sizes[0]/sizes)   # SE(500)*sqrt(500) = SD(100000)
        ax1.plot(sizes, ref_point, 'r--',  alpha=0.5, label=r'$\frac{SD}{\sqrt{ss}}$')
        ax1.legend()

        # plot bias
        ax2.errorbar(sizes, biases, yerr=uncertainties/np.sqrt(self.n_ensemble), fmt='o-', capsize=5)         
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Bias on λ')
        ax2.set_title('Bias on λ as a function of the sample size')

        # add data details
        for i, (x, y) in enumerate(zip(sizes, biases)):
            if i == len(sizes)-1:
                ax2.annotate(f'({x}, {y:.4f})', xy=(x, y), xytext=(-50, -30), textcoords='offset points', fontsize=8)
            else:     
                ax2.annotate(f'({x}, {y:.4f})', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
                        
                        
                        
                        

        plt.tight_layout()
        return fig

