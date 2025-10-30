import numpy as np
from scipy.stats import norm
import pandas as pd
from dataclasses import dataclass

@dataclass
class Option:
    S0: float  # Current price of underlying
    K: float  # Strike price
    T: float  # Time to maturity (in years)
    r: float  # Risk-free rate
    sigma: float  # Volatility (std dev per year)
    o_type: str = "call"  # "call" or "put"
    q: float = 0.0 # Continuous dividend yield
    def __post_init__(self):
        if self.o_type not in ['call', 'put']:
            raise ValueError("Invalid Value for Option Type [o_type]")

    def payoff(self, ST):
        if self.o_type == "call":
            return max(ST - self.K, 0)
        else:
            return max(self.K - ST, 0)

    def d1_d2(self):
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.q - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return d1, d2

    def black_scholes(self):
        d1, d2 = self.d1_d2()
        if self.o_type == "call":
            price = self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.o_type == "put":
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)

        return price
