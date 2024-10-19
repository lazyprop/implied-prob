import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from black_scholes import construct_distribution
import itertools
import numpy as np

def collect_option_prices(ticker, expiration):
    stock = yf.Ticker(ticker)
    option_chain = stock.option_chain(expiration)
    calls = option_chain.calls
    option_data = []
    for index, row in calls.iterrows():
        option_data.append({
            'strike': row['strike'],
            'lastPrice': row['lastPrice'],
            'expiration': expiration,
            'daysToExpiration': (pd.to_datetime(expiration) - pd.to_datetime('today')).days
        })
    return option_data

def get_current_price(ticker):
    return yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]

def plot_distributions(ticker, expirations, S0):
    colors = itertools.cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])
    plt.figure(figsize=(12, 8))
    
    for expiration in expirations:
        option_data = collect_option_prices(ticker, expiration)
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
    plt.title(f'Implied Probability Density of {ticker}')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_ticker(ticker):
    S0 = get_current_price(ticker)
    stock = yf.Ticker(ticker)
    expirations = stock.options[:5]
    plot_distributions(ticker, expirations, S0)

analyze_ticker('AAPL')
