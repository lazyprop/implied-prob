from collections import namedtuple
from math import log, exp, sqrt, erf
from scipy.optimize import brentq
import numpy as np

Option = namedtuple('Option', ['S0', 'X', 'T_days', 'r', 'sigma'])

def cum_normal(x):
    return (1 + erf(x / sqrt(2))) / 2

def call_price(option):
    T = option.T_days / 365
    d1 = (log(option.S0 / option.X) + (option.r + 0.5 * option.sigma**2) * T) / (option.sigma * sqrt(T))
    d2 = d1 - option.sigma * sqrt(T)
    return option.S0 * cum_normal(d1) - option.X * exp(-option.r * T) * cum_normal(d2)

def implied_vol(option, market_price):
    def obj_func(sigma):
        return call_price(option._replace(sigma=sigma)) - market_price

    try:
        return brentq(obj_func, 0.0001, 3.0)
    except ValueError:
        return None

def construct_distribution(option_data, S0, r):
    strikes = []
    implied_vols = []
    
    for option in option_data:
        market_price = option['lastPrice']
        strike = option['strike']
        T_days = option['daysToExpiration']
        iv = implied_vol(Option(S0, strike, T_days, r, 0.2), market_price)
        
        if iv is not None:
            strikes.append(strike)
            implied_vols.append(iv)

    return {X: (1 / (X * sigma * sqrt(2 * np.pi))) * exp(-0.5 * (log(X) - log(S0))**2 / sigma**2) for X, sigma in zip(strikes, implied_vols)}

def plot_distribution(implied_pdf):
    plt.figure(figsize=(10, 6))
    plt.plot(implied_pdf.keys(), implied_pdf.values(), label='Implied Probability Density', marker='o')
    plt.title('Implied Probability Density of AAPL Stock')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.show()
