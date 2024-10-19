import yfinance as yf
import pandas as pd

def fetch_option_prices(ticker, expiration):
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

def fetch_current_price(ticker):
    return yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]

def fetch_expirations(ticker):
    return yf.Ticker(ticker).options[:5]
