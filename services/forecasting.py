"""
Time-series forecasting models for hospital resource optimization.
Includes ARIMA, Prophet, and a lightweight LSTM model for patient prediction.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logs

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred):
    """Calculates and returns standard time-series error metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred))
    }

def forecast_arima(df, target_col='patients', forecast_days=7):
    """ARIMA baseline forecasting model."""
    logger.info("Running ARIMA forecasting...")
    try:
        train = df[target_col].iloc[:-forecast_days]
        test = df[target_col].iloc[-forecast_days:]
        
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        test_pred = model_fit.forecast(steps=forecast_days)
        metrics = evaluate_model(test, test_pred)
        
        full_model = ARIMA(df[target_col], order=(5, 1, 0))
        full_model_fit = full_model.fit()
        future_pred = full_model_fit.forecast(steps=forecast_days)
        
        return {"name": "ARIMA", "metrics": metrics, "predictions": future_pred.tolist()}
    except Exception as e:
        logger.error(f"ARIMA error: {e}")
        return None

def forecast_prophet(df, target_col='patients', forecast_days=7):
    """Prophet forecasting model handling weekly seasonality well."""
    logger.info("Running Prophet forecasting...")
    try:
        prophet_df = df[['date', target_col]].copy()
        prophet_df['ds'] = pd.to_datetime(prophet_df['date'])
        prophet_df['y'] = prophet_df[target_col]
        train = prophet_df.iloc[:-forecast_days]
        test = prophet_df.iloc[-forecast_days:]
        
        m = Prophet(daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.05, seasonality_prior_scale=5)
        m.fit(train)
        
        future_test = m.make_future_dataframe(periods=forecast_days)
        forecast_test = m.predict(future_test)
        test_pred = forecast_test['yhat'].iloc[-forecast_days:]
        metrics = evaluate_model(test['y'], test_pred)
        
        m_full = Prophet(daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.05, seasonality_prior_scale=5)
        m_full.fit(prophet_df)
        future = m_full.make_future_dataframe(periods=forecast_days)
        forecast = m_full.predict(future)
        future_pred = forecast['yhat'].iloc[-forecast_days:].values
        
        future_pred = np.maximum(0, future_pred)
        return {"name": "Prophet", "metrics": metrics, "predictions": future_pred.tolist()}
    except Exception as e:
        logger.error(f"Prophet error: {e}")
        return None

def forecast_lstm(df, target_col='patients', forecast_days=7):
    """Lightweight LSTM deep learning model using TensorFlow."""
    logger.info("Running LSTM forecasting...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Input
        from sklearn.preprocessing import MinMaxScaler
        
        # Performance Setting: Run in CPU fast mode to ensure quick dashboard loads
        tf.config.set_visible_devices([], 'GPU')
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[[target_col]].values)
        
        def create_dataset(dataset, time_step=14):
            X, Y = [], []
            for i in range(len(dataset)-time_step-1):
                X.append(dataset[i:(i+time_step), 0])
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)
            
        time_step = 14
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        split = len(X) - forecast_days
        if split < 0:
            raise ValueError("Not enough data to split for LSTM.")
            
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = Sequential()
        model.add(Input(shape=(time_step, 1)))
        model.add(LSTM(16, activation='relu', return_sequences=False)) # Lightweight size
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Extremely fast training for UI responsiveness
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        
        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = scaler.inverse_transform(test_pred_scaled).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        min_len = min(len(test_pred), forecast_days)
        metrics = evaluate_model(y_test_inv[-min_len:], test_pred[-min_len:])
        
        last_sequence = scaled_data[-time_step:]
        future_preds = []
        for _ in range(forecast_days):
            seq = last_sequence.reshape(1, time_step, 1)
            next_pred = model.predict(seq, verbose=0)[0][0]
            future_preds.append(next_pred)
            last_sequence = np.append(last_sequence[1:], [next_pred])
            
        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
        future_preds = np.maximum(0, future_preds)
        
        return {"name": "LSTM", "metrics": metrics, "predictions": future_preds.tolist()}
    except ImportError:
        logger.warning("TensorFlow not installed. Skipping LSTM model.")
        return None
    except Exception as e:
        logger.error(f"LSTM error: {e}")
        return None

def get_all_forecasts(df, target_col='patients', forecast_days=7):
    """
    Runs all models and ranks them.
    Returns:
        list[dict]: List of models with predictions, metrics, and is_best flag.
    """
    models = []
    
    # Run sequentially
    for func in [forecast_arima, forecast_prophet, forecast_lstm]:
        res = func(df, target_col, forecast_days)
        if res:
            res['is_best'] = False
            models.append(res)
            
    if not models:
        raise ValueError("All models failed to generate predictions.")
        
    # Find Best Model (Min MAPE)
    best_model = min(models, key=lambda x: x['metrics']['MAPE'])
    best_model['is_best'] = True
    
    return models

def get_best_forecast(df, target_col='patients', forecast_days=7):
    """Wrapper function to return just the best performing model."""
    models = get_all_forecasts(df, target_col, forecast_days)
    for m in models:
        if m['is_best']: return m
    return models[0]
