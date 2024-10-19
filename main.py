import itertools
import numpy as np
import matplotlib.pyplot as plt
from fetch import fetch_option_prices, fetch_current_price, fetch_expirations
from black_scholes import construct_distribution

def plot_distributions(ticker, expirations, S0):
    colors = itertools.cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])
    plt.figure(figsize=(12, 8))
    for expiration in expirations:
        option_data = fetch_option_prices(ticker, expiration)
        if not option_data:
            continue
        implied_pdf = construct_distribution(option_data, S0, 0.05)
        strikes = np.array(list(implied_pdf.keys()))
        densities = np.array(list(implied_pdf.values()))
        if strikes.size == 0 or densities.size == 0:
            continue
        strike_range = np.linspace(strikes.min(), strikes.max(), 100)
        density_interp = np.interp(strike_range, strikes, densities)
        color = next(colors)
        plt.plot(strike_range, density_interp, label=f'Expiration: {expiration}', color=color)
    plt.axvline(S0, color='black', linestyle='--', label='Current Price')
    plt.title(f'Implied Probability Densities of {ticker}')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_ticker(ticker):
    S0 = fetch_current_price(ticker)
    expirations = fetch_expirations(ticker)
    plot_distributions(ticker, expirations, S0)

analyze_ticker('AAPL')
