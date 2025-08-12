import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def load_financial_data(tickers=['TSLA', 'BND', 'SPY'], 
                      start_date='2015-08-11',
                      end_date='2025-08-11',
                      interval='1d',
                      data_dir='../../data/raw'):
    """
    Fetch financial data from Yahoo Finance and save to CSV files
    Returns merged DataFrame of closing prices
    """
    os.makedirs(data_dir, exist_ok=True)
    
    dfs = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        df.to_csv(f'{data_dir}/{ticker}_data.csv')
        close_series = df['Close']
        close_series.name = ticker
        dfs.append(close_series)
    
    return pd.concat(dfs, axis=1)