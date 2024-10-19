from collections import namedtuple
from math import log, exp, sqrt
from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

Opt = namedtuple('Opt', ['S0', 'X', 'T', 'r', 'sigma'])

def call_price(opt):
    T = opt.T / 365
    d1 = (log(opt.S0 / opt.X) + (opt.r + 0.5 * opt.sigma**2) * T) / (opt.sigma * sqrt(T))
    d2 = d1 - opt.sigma * sqrt(T)
    return opt.S0 * norm.cdf(d1) - opt.X * exp(-opt.r * T) * norm.cdf(d2)

def put_price(opt):
    T = opt.T / 365
    d1 = (log(opt.S0 / opt.X) + (opt.r + 0.5 * opt.sigma**2) * T) / (opt.sigma * sqrt(T))
    d2 = d1 - opt.sigma * sqrt(T)
    return opt.X * exp(-opt.r * T) * norm.cdf(-d2) - opt.S0 * norm.cdf(-d1)

def implied_vol(opt, market_price):
    def obj_func(sigma):
        return call_price(opt._replace(sigma=sigma)) - market_price
    try:
        return brentq(obj_func, 0.0001, 3.0)
    except ValueError:
        return None

def norm_dist(dist):
    total_area = sum(dist.values())
    return {k: v / total_area for k, v in dist.items()}

def construct_dist(opt_data, S0, r):
    strikes, ivs = [], []
    for opt in opt_data:
        market_price = opt['lastPrice']
        strike = opt['strike']
        T_days = opt['daysToExpiration']
        iv = implied_vol(Opt(S0, strike, T_days, r, 0.2), market_price)
        if iv is not None:
            strikes.append(strike)
            ivs.append(iv)
    dist = {X: (1 / (X * sigma * sqrt(2 * np.pi))) * exp(-0.5 * (log(X) - log(S0))**2 / sigma**2)
            for X, sigma in zip(strikes, ivs)}
    return norm_dist(dist)

def plot_dist(implied_pdf):
    plt.figure(figsize=(10, 6))
    plt.plot(implied_pdf.keys(), implied_pdf.values(), label='Implied Probability Density', marker='o')
    plt.title('Implied Probability Density of Stock')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.show()
