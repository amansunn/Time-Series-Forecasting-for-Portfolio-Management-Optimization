import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_financial_data(df):
    """
    Clean and preprocess raw financial data:
    - Handle missing values using forward fill
    - Remove duplicate entries
    - Ensure datetime index
    """
    df = df.asfreq('D').ffill()
    df = df[~df.index.duplicated(keep='first')]
    return df.dropna()

def calculate_returns(df):
    """Calculate daily percentage returns and log returns"""
    returns = df.pct_change().dropna()
    log_returns = np.log(df/df.shift(1)).dropna()
    return returns, log_returns

def add_volatility_metrics(df, window=30):
    """
    Add volatility metrics:
    - Rolling standard deviation
    - Bollinger Bands
    - Exponential moving average
    """
    df = df.copy()
    rolling = df.rolling(window=window)
    
    df['Rolling_Mean'] = rolling.mean()
    df['Rolling_Std'] = rolling.std()
    df['Upper_Bollinger'] = df['Rolling_Mean'] + (df['Rolling_Std'] * 2)
    df['Lower_Bollinger'] = df['Rolling_Mean'] - (df['Rolling_Std'] * 2)
    df['EMA'] = df.ewm(span=window, adjust=False).mean()
    
    return df.dropna()

def prepare_dataset(raw_df):
    """Full preprocessing pipeline"""
    cleaned_df = clean_financial_data(raw_df)
    returns_df, log_returns_df = calculate_returns(cleaned_df)
    volatility_df = add_volatility_metrics(cleaned_df)
    
    full_df = pd.concat([cleaned_df, returns_df.add_prefix('Returns_'), 
                       log_returns_df.add_prefix('LogReturns_'),
                       volatility_df.add_prefix('Volatility_')], axis=1)
    
    return full_df.dropna()