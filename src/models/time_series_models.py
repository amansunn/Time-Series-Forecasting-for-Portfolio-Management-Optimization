import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

class ARIMAModel:
    def __init__(self, ticker='TSLA'):
        self.ticker = ticker
        self.model = None
        
    def train(self, train_data, seasonal=False):
        if seasonal:
            self.model = auto_arima(train_data,
                                  seasonal=True,
                                  m=12,
                                  trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True)
        else:
            self.model = auto_arima(train_data,
                                  trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True)
        return self.model
        
    def forecast(self, steps, alpha=0.05):
        """Generate forecast with confidence intervals"""
        forecast = self.model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
        return forecast[0], forecast[1]
    
    @staticmethod
    def calculate_metrics(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

class LSTMModel:
    def __init__(self, n_steps=60, units=50, epochs=50, batch_size=32):
        self.n_steps = n_steps
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
    def train(self, train_data):
        # Prepare data for LSTM
        X, y = create_sequences(train_data.values, self.n_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.n_steps, 1)),
            LSTM(self.units),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='mean_squared_error')
        
        history = self.model.fit(X, y, epochs=self.epochs, 
                               batch_size=self.batch_size, verbose=1)
        return history
    
    def forecast(self, test_data):
        # Create sequences from test data and predict
        X_test, y_test = create_sequences(test_data.values, self.n_steps)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return self.model.predict(X_test)