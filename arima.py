#ARIMA Model for Forecasting (for comparison)
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from load_dataset import load_data 
from sklearn.metrics import mean_squared_error, mean_absolute_error


def arima_forecast(ticker, start_date, end_date, order=(1,1,1)):
    # Load data
    data = load_data(ticker, start_date, end_date)
    
    # Prepare data for ARIMA
    prices = data['Close'].values
    
    # Fit ARIMA model
    model = ARIMA(prices, order=order)
    fitted_model = model.fit()
    
    # Forecast next 30 days
    forecast = fitted_model.forecast(steps=30)
    
    return forecast


# EVALUATION METRICS
def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    theil_u = np.sqrt(np.mean((predicted - actual) ** 2)) / (np.sqrt(np.mean(predicted ** 2)) + np.sqrt(np.mean(actual ** 2)))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Theil_U': theil_u
    }
