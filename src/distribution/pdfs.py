import numpy as np
from scipy import stats
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PDF:
    def __init__(self, mu=3, sigma=0.3,  beta=1, m=1.4, f=0.6, lamda=0.3, mu_b=0, sigma_b=2.5):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.m = m
        self.f = f
        self.lamda = lamda
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.x_min, self.x_max = 0, 5
        self.y_min, self.y_max = 0, 10
        
    def g_s(self, x):
        """ Truncated Crystal Ball distribution for signal in X."""

        # mask for the interval limits
        mask = (x >= self.x_min) & (x <= self.x_max)
        
        # initialise the pdf values
        pdf = np.zeros_like(x, dtype=float)
        
        # normalisation constant
        N_inverse = (stats.crystalball.cdf(self.x_max, self.beta, self.m, loc=self.mu, scale=self.sigma) -
                     stats.crystalball.cdf(self.x_min, self.beta, self.m, loc=self.mu, scale=self.sigma))

        # calculate pdf for values within the valid interval
        valid_x = x[mask]
        
        pdf[mask] = stats.crystalball.pdf(valid_x, self.beta, self.m, loc=self.mu, scale=self.sigma) / N_inverse
                                        
        return pdf
    
    def h_s(self, y):
        """ Truncated exponential distribution for signal in Y."""

        mask = (y >= self.y_min) & (y <= self.y_max)
        pdf = np.zeros_like(y, dtype=float)
        
        # (inverse) normalisation constant 
        N_inverse = (1 - np.exp(-self.lamda * self.y_max))
        pdf[mask] = self.lamda * np.exp(-self.lamda * y[mask]) / N_inverse
        
        return pdf
    
    def g_b(self, x):
        """ Uniform distribution for background in X.""" 

        mask = (x >= self.x_min) & (x <= self.x_max)
        pdf = np.zeros_like(x, dtype=float)
        pdf[mask] = 1 / (self.x_max - self.x_min)

        return pdf
    
    def h_b(self, y):
        """ Truncated normal distribution for background in Y."""
        mask = (y >= self.y_min) & (y <= self.y_max)
        pdf = np.zeros_like(y, dtype=float)
        
        # (inverse) normalisation constant
        N_inverse = (stats.norm.cdf(self.y_max, self.mu_b, self.sigma_b) - 
                     stats.norm.cdf(self.y_min, self.mu_b, self.sigma_b))

        pdf[mask] = (stats.norm.pdf(y[mask], self.mu_b, self.sigma_b) / N_inverse)
        
        return pdf
    
    def s_pdf(self, x, y):
        """ Joint pdf for signal component: gs(X)hs(Y)."""
        
        return self.g_s(x) * self.h_s(y)
    
    def b_pdf(self, x, y):
        """ Joint PDF for background component: gb(X)hb(Y)."""
        
        return self.g_b(x) * self.h_b(y)
    
    def mix_pdf(self, x, y):
        """ Full mix model: f*s(X,Y) + (1-f)*b(X,Y)."""
          
        return (self.f * self.s_pdf(x, y) + (1 - self.f) * self.b_pdf(x, y))
    
    def verify(self):
        """ Verify all p.d.f.s are nomalised over the valid domains."""

        def integrate_1d(pdf_func, min, max):
            result, error = quad(lambda x: pdf_func(np.array([x]))[0], min, max)
            return result
        
        def integrate_2d(pdf_func):
            result, error = dblquad(
                lambda y, x: pdf_func(np.array([x]), np.array([y]))[0],
                self.x_min, self.x_max,
                lambda x: self.y_min, lambda x: self.y_max)
            return result
        
        # veriy g_s(x)
        gs_integral = integrate_1d(self.g_s, 0, 5)
        print(f"g_s(x) pdf integral: {gs_integral:.4f}")
        
        # veriy h_s(y)
        hs_integral = integrate_1d(self.h_s, 0, 10)
        print(f"h_s(y) pdf integral: {hs_integral:.4f}")

        # veriy g_b(x)
        gb_integral = integrate_1d(self.g_b, 0, 5)
        print(f"g_b(x) pdf integral: {gb_integral:.4f}")
        
        # veriy h_b(y)
        hb_integral = integrate_1d(self.h_b, 0, 10)
        print(f"h_b(y) pdf integral: {hb_integral:.4f}")

        # veriy s(x, y)
        s_integral = integrate_2d(self.s_pdf)
        print(f"signal pdf integral: {s_integral:.4f}")
        
        # veriy b(x, y)
        b_integral = integrate_2d(self.b_pdf)
        print(f"background integral: {b_integral:.4f}")
        
        # veriy mixed f(x, y)
        mix_integral = integrate_2d(self.mix_pdf)
        print(f"mixted pdf integral: {mix_integral:.4f}")

    def plot_x_pdf(self, ax):
        """ Plot the projection in X-axis."""

        x = np.linspace(self.x_min, self.x_max, 200)
        # y_mid = (self.y_min + self.y_max) / 2
        
        # marginal distributions
        signal_x = self.f * self.g_s(x)
        background_x = (1 - self.f) * self.g_b(x)
        total_x = signal_x + background_x
        
        ax.plot(x, total_x, 'k-', label='Total p.d.f', linewidth=2)
        ax.plot(x, signal_x, 'b--', label='Signal')
        ax.plot(x, background_x, 'r-.', label='Background')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Probability Density')
        ax.set_title('Projection in X')
        ax.legend()
        ax.grid(True, alpha=0.2)

    def plot_y_pdf(self, ax):
        """ Plot the projection in Y-axis."""

        y = np.linspace(self.y_min, self.y_max, 200)
        # x_mid = (self.x_min + self.x_max) / 2
        
        # marginal distributions
        signal_y = self.f * self.h_s(y)
        background_y = (1 - self.f) * self.h_b(y)
        total_y = signal_y + background_y

        ax.plot(y, total_y, 'k-', label='Total p.d.f', linewidth=2)
        ax.plot(y, signal_y, 'b--', label='Signal')
        ax.plot(y, background_y, 'r-.', label='Background')
        
        ax.set_xlabel('Y')
        ax.set_ylabel('Probability Density')
        ax.set_title('Projection in Y')
        ax.legend()
        ax.grid(True, alpha=0.2)

    def plot_joint_pdf(self, ax):
        """ Create a 2D plot of the joint probability density."""

        # build the grid
        x = np.linspace(self.x_min, self.x_max, 100)
        y = np.linspace(self.y_min, self.y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # put the joint p.d. values on the grid
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j,i] = self.mix_pdf(np.array([X[j,i]]), np.array([Y[j,i]]))
                                        
        # create the colourmap plot
        im = ax.pcolormesh(X, Y, Z, cmap='YlGnBu')
        
        # add a colourbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Probability Density')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Joint Probability Density')

    def plot(self):
        """ Set the subplots."""
        
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131)
        self.plot_x_pdf(ax1)
        
        ax2 = fig.add_subplot(132)
        self.plot_y_pdf(ax2)
        
        ax3 = fig.add_subplot(133)
        self.plot_joint_pdf(ax3)
        
        plt.tight_layout()
        
        return fig