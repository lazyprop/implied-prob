import yfinance as yf
import itertools
import numpy as np
import matplotlib.pyplot as plt
from fetch import fetch_option_data, fetch_current_price, fetch_expirations
from black_scholes import construct_dist
from scipy.integrate import quad
from scipy.optimize import curve_fit

def gauss(x, mean, stddev, amp):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def fit_gauss(strikes, dens):
    mean = np.sum(strikes * dens) / np.sum(dens)
    stddev = np.sqrt(np.sum(dens * (strikes - mean) ** 2) / np.sum(dens))
    popt, _ = curve_fit(gauss, strikes, dens, p0=[mean, stddev, max(dens)])
    return popt

def interp_norm(dist):
    strikes = np.array(list(dist.keys()))
    dens = np.array(list(dist.values()))
    mean, stddev, amp = fit_gauss(strikes, dens)
    interp_func = lambda x: gauss(x, mean, stddev, amp)
    x_range = np.linspace(strikes.min(), strikes.max(), 100)
    norm_factor = np.trapezoid(interp_func(x_range), x_range)
    return lambda x: interp_func(x) / norm_factor

def plot_dists(dists, interp_dists, expiries, ticker):
    colors = itertools.cycle(['blue', 'orange', 'green', 'red', 'purple'])
    plt.figure(figsize=(12, 8))
    strike_range = np.linspace(min([min(list(d.keys())) for d in dists]),
                               max([max(list(d.keys())) for d in dists]), 100)
    for interp_dist, expiry in zip(interp_dists, expiries):
        plt.plot(strike_range, interp_dist(strike_range), label=f'Expiration: {expiry}', color=next(colors))
    plt.axvline(fetch_current_price(ticker), color='black', linestyle='--', label='Current Price')
    plt.title(f'Implied Probability Densities of {ticker}')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.show()

def integrate_ipdf(interp_dist):
    result, _ = quad(interp_dist, 0, np.inf)
    return result

def analyze_ticker(ticker):
    S0 = fetch_current_price(ticker)
    expiries = fetch_expirations(ticker)
    dists = []
    interp_dists = []
    for expiry in expiries:
        opt_data = fetch_option_data(ticker, expiry)
        if not opt_data:
            continue
        implied_pdf = construct_dist(opt_data, S0, 0.05)
        dists.append(implied_pdf)
        interp_dist = interp_norm(implied_pdf)
        interp_dists.append(interp_dist)
        print(f'Integrated Probability for {expiry}: {integrate_ipdf(interp_dist)}')
    plot_dists(dists, interp_dists, expiries, ticker)

analyze_ticker('^NSEI')
