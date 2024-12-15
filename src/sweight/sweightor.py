from dataclasses import dataclass
import numpy as np
from scipy import stats
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from sweights import SWeight
import matplotlib.pyplot as plt


@dataclass
class SweightsResult:
    """Store results from sWeight."""
    lambda_est: float
    lambda_err: float
    bias: float
    mi_x: Minuit
    sweight: np.ndarray
    bweight: np.ndarray

class Sweightor:
    """A class for performing sWeight."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N_x = len(x)
        self.x_range = (0, 5)
        self.y_range = (0, 10)
        
    def EMLx_fit(self):
        """Do the extended maximum likelihood for X only."""

        def density(xv, N, mu, sigma, beta, m, f):
            norm_s = (stats.crystalball.cdf(5, beta, m, loc=mu, scale=sigma) - stats.crystalball.cdf(0, beta, m, loc=mu, scale=sigma))
            signal = stats.crystalball.pdf(xv, beta, m, loc=mu, scale=sigma) / norm_s
            
            background = 1 / 5
            
            pdf = f * signal + (1-f) * background

            return N, N*pdf

        nll = ExtendedUnbinnedNLL(self.x, density)

        mi = Minuit(nll, mu=3.0, sigma=0.3, beta=1.0, m=1.4, f=0.6, N=self.N_x)

        mi.migrad()
        mi.hesse()  
        
        # define pdfs with optimal params
        def signal_pdf(x):
            norm = (stats.crystalball.cdf(5, mi.values['beta'], mi.values['m'],loc=mi.values['mu'], scale=mi.values['sigma']) -                    
                    stats.crystalball.cdf(0, mi.values['beta'], mi.values['m'], loc=mi.values['mu'], scale=mi.values['sigma']))
                                        
            return stats.crystalball.pdf(x, mi.values['beta'], mi.values['m'], loc=mi.values['mu'], scale=mi.values['sigma']) / norm
                                       
        def background_pdf(x):
            return np.ones_like(x) / 5.0
        
        return mi, signal_pdf, background_pdf
    
    def est_lambda(self, weights):
        """ Estimate lambda using weighted data."""
        def weighted_nll(lamda):
            # y = np.linspace(*self.y_range, 200)
            pdf = lamda * np.exp(-lamda * self.y) / (1 - np.exp(-10 * lamda))
            return -np.sum(weights * np.log(pdf))
        
        # fit for lambda
        m = Minuit(weighted_nll, lamda=0.3)
        m.limits['lamda'] = (0.1, 1.0)
        m.migrad()
        m.hesse()     
        return m.values['lamda'], m.errors['lamda']
    
    def do_sWeight(self, true_lambda = 0.3):
        """
        Perform sWeights analysis.
        
        Parameters:
            true_lambda: true value of lambda for bias calculation    
        Returns:
            SweightsResult containing fit results and analysis
        """
        # fit X distribution
        mi, spdf, bpdf = self.EMLx_fit()
        
        # use optimal f
        Ns = mi.values['N'] * mi.values['f']
        Nb = mi.values['N'] * (1 - mi.values['f'])
        
        # sWeights
        sweighter = SWeight(self.x, pdfs=[spdf, bpdf], yields=[Ns, Nb], discvarranges=([0, 5],))
        sw = sweighter.get_weight(0, self.x)
        bw = sweighter.get_weight(1, self.x)
        
        # estimate lambda
        lambda_est, lambda_err = self.est_lambda(sw)
        
        return SweightsResult(
            lambda_est=lambda_est,
            lambda_err=lambda_err,
            bias=lambda_est - true_lambda,
            mi_x=mi,
            sweight=sw,
            bweight=bw)

    def plot_results(self, result: SweightsResult, true_params):
        """
        Plot the 'sweight+bweight' check, weighted result in Y and true distribution of Y.
        
        Parameters:
            result: all results stored in SweightsResult
            true_params: true value of distribution for ploting the true distributions of Y   
        """
        sw = result.sweight
        bw = result.bweight
        N_total = len(self.y)
        f = np.sum(sw) / N_total  
        tlambda = true_params[5]
        tmu_b = true_params[6]
        tsigma_b = true_params[7]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ------------------------- plot 1: check weigts' sum ----------------------------- #
        mi, spdf, bpdf = self.EMLx_fit()
        Ns = mi.values['N'] * mi.values['f']
        Nb = mi.values['N'] * (1 - mi.values['f'])
        
        sweighter = SWeight(self.x, pdfs=[spdf, bpdf], yields=[Ns, Nb],discvarranges=([0, 5],))

        # get uniformly distributed weights!
        x = np.linspace(*self.x_range, 200)
        sw0 = sweighter.get_weight(0, x)
        bw0 = sweighter.get_weight(1, x)
        ax1.plot(x, sw0, 'b--', label='Signal Weight')
        ax1.plot(x, bw0, 'r-.', label='Background Weight')
        ax1.plot(x, sw0+bw0, 'k-', label='Total Weight')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Weights')
        ax1.set_title('Weight Functions')
        ax1.legend()

        # ------------------------------ plot 2: sweighted in Y ---------------------------------- #
        y_y, x_y = np.histogram(self.y, bins=50, weights=sw, range=self.y_range, density=True)
        center_y = (x_y[:-1] + x_y[1:]) / 2
        y_y *= f
        bin_width = x_y[1] - x_y[0]
        N_eff = np.sum(sw)  
        error_y = np.sqrt(np.histogram(self.y, bins=50, weights=sw**2, range=self.y_range, density=False)[0])
        error_y = error_y / (N_eff * bin_width)
        error_y *= f 
        ax2.errorbar(center_y, y_y, yerr=error_y, fmt='o', label='sWeighted Data')

        # ------------------------------ plot 2: bweighted in Y ---------------------------------- #
        y_by, x_by = np.histogram(self.y, bins=50, weights=bw, range=self.y_range, density=True)
        center_by = (x_by[:-1] + x_by[1:]) / 2
        y_by *= (1-f)
        bbin_width = x_by[1] - x_by[0]
        bN_eff = np.sum(bw)
        error_by = np.sqrt(np.histogram(self.y, bins=50, weights=bw**2, range=self.y_range, density=False)[0])
        error_by = error_by / (bN_eff * bbin_width)
        error_by *= (1-f) 
        ax2.errorbar(center_by, y_by, yerr=error_by, fmt='o', label='bWeighted Data')

        # ---------------------------- plot 2: the true distribution ----------------------------- #
        y_plot = np.linspace(0, 10, 50)
        # true signal
        true_signal = tlambda * np.exp(-tlambda * y_plot)
        true_signal /= (1 - np.exp(-10 * tlambda))
        true_signal *= f
        ax2.hist(y_plot, bins=50, weights=true_signal, 
                histtype='step', label='True Signal', color='g')
        
        # true background
        true_bg = stats.norm.pdf(y_plot, tmu_b, tsigma_b)
        true_bg /= (stats.norm.cdf(10, tmu_b, tsigma_b) - 
                    stats.norm.cdf(0, tmu_b, tsigma_b))
        true_bg *= (1-f)
        ax2.hist(y_plot, bins=50, weights=true_bg,
                histtype='step', label='True Background', color='r')
        
        # ------------------------------------------------------------------------------------ #
        ax2.set_xlim(0, 10)
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Signal Density in Y')
        ax2.set_title('Weighted Signal Component in Y')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


