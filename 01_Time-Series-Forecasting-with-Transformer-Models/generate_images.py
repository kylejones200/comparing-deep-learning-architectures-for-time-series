#!/usr/bin/env python3
"""Generate Tufte-style visualizations for transformer forecasting article."""

import signalplot
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Set random seeds
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    tf = None
except Exception:
    tf = None

signalplot.apply(font_family='serif')

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

original_savefig = plt.savefig


def savefig_tufte(filename: str | Path, **kwargs) -> None:
    """Save figures in the `images` directory with consistent style."""
    filename_str = str(filename)
    if not filename_str.startswith("/") and not filename_str.startswith("images/"):
        filename = IMAGES_DIR / filename_str
    original_savefig(filename, **kwargs)
    logger.info("Saved figure to %s", filename)


plt.savefig = savefig_tufte


# Set random seeds for reproducibility
try:
    tf.random.set_seed(42)
except ImportError:
    tf = None

# Load Oklahoma energy consumption data
data_path = Path("../../geospatial/datasets/use_OK.csv")
df = pd.read_csv(data_path)

# The dataset has multiple MSN (Metric Serial Number) codes
use_cols = [col for col in df.columns if col.isdigit()]

# Melt the dataframe to long format
df_long = df.melt(
    id_vars=['State', 'MSN'],
    value_vars=use_cols,
    var_name='Year',
    value_name='Value'
)

df_long['Year'] = pd.to_datetime(df_long['Year'], format='%Y')
df_long = df_long.sort_values('Year')

# Filter for total energy consumption metrics
# MSN codes with 'TOT' typically represent total consumption
total_energy = df_long[df_long['MSN'].str.contains('TOT', na=False)].copy()

# Aggregate by year (sum across all total consumption metrics)
total_energy = total_energy.groupby('Year')['Value'].sum().reset_index()
total_energy = total_energy[total_energy['Value'].notna() & (total_energy['Value'] > 0)]

ts_data = total_energy.set_index('Year')['Value']
ts_data = ts_data.sort_index()

logger.info(f"Time series length: {len(ts_data)}")
logger.info(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
logger.info(f"Value range: {ts_data.min():.2f} to {ts_data.max():.2f}")
logger.info(f"\nFirst 10 values:\n{ts_data.head(10)}")



# Code block 2
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts_data.values.reshape(-1, 1)).flatten()

def create_sequences(data, seq_length, forecast_horizon=12):
    """
    Create input-output sequences for time series forecasting.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    seq_length : int
        Number of past observations to use as input
    forecast_horizon : int
        Number of future periods to forecast
    
    Returns:
    --------
    X : ndarray
        Input sequences (n_samples, seq_length)
    y : ndarray
        Target sequences (n_samples, forecast_horizon)
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(X), np.array(y)

# Configuration
seq_length = 20  # Use 20 years of history
forecast_horizon = 5  # Forecast 5 years ahead

X, y = create_sequences(ts_scaled, seq_length, forecast_horizon)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

logger.info(f"Training samples: {len(X_train)}")
logger.info(f"Test samples: {len(X_test)}")
logger.info(f"Sequence length: {seq_length}")
logger.info(f"Forecast horizon: {forecast_horizon}")



# Code block 3
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# Use original (non-scaled) data for ARIMA
ts_train = ts_data[:split_idx + seq_length]
ts_test = ts_data[split_idx + seq_length:]

# Fit ARIMA model
# Order (p, d, q): p=autoregressive, d=differencing, q=moving average
logger.info("Fitting ARIMA model...")
start_time = time.time()

arima_model = ARIMA(ts_train, order=(2, 1, 2))
arima_fitted = arima_model.fit()

arima_time = time.time() - start_time
logger.info(f"ARIMA training time: {arima_time:.2f} seconds")

# Forecast
arima_forecast = arima_fitted.forecast(steps=min(len(ts_test), forecast_horizon))

arima_mae = mean_absolute_error(ts_test[:len(arima_forecast)], arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(ts_test[:len(arima_forecast)], arima_forecast))

logger.info(f"ARIMA MAE: {arima_mae:.2f}")
logger.info(f"ARIMA RMSE: {arima_rmse:.2f}")



# Code block 4
try:
except ImportError:
    tf = None
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(forecast_horizon)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

# Train with early stopping
logger.info("Training LSTM model...")
start_time = time.time()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

lstm_time = time.time() - start_time
logger.info(f"LSTM training time: {lstm_time:.2f} seconds")

lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)

lstm_mae = mean_absolute_error(y_test[:, 0], lstm_pred[:, 0])
lstm_rmse = np.sqrt(mean_squared_error(y_test[:, 0], lstm_pred[:, 0]))

logger.info(f"LSTM MAE: {lstm_mae:.4f}")
logger.info(f"LSTM RMSE: {lstm_rmse:.4f}")



# Code block 5
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Input, Dropout
from tensorflow.keras.models import Model
np.random.seed(42)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer encoder block with multi-head attention"""
    # Multi-head self-attention
    attention = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # Feed-forward network
    ffn = Dense(ff_dim, activation="relu")(attention)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)
    outputs = LayerNormalization(epsilon=1e-6)(attention + ffn)
    return outputs

def build_transformer_model(seq_length, forecast_horizon, head_size=64, 
                           num_heads=4, ff_dim=128, num_layers=2, dropout=0.2):
    """Build Transformer model for time series forecasting"""
    inputs = Input(shape=(seq_length, 1))
    x = inputs
    
    # Stack transformer encoder layers
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Global average pooling to get fixed-size representation
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(forecast_horizon)(x)
    
    model = Model(inputs, outputs)
    return model

# Build and compile model
transformer_model = build_transformer_model(seq_length, forecast_horizon)
transformer_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
transformer_model.summary()

# Train
logger.info("Training Transformer model...")
start_time = time.time()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_trans = transformer_model.fit(
    X_train_lstm, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

transformer_time = time.time() - start_time
logger.info(f"Transformer training time: {transformer_time:.2f} seconds")

transformer_pred = transformer_model.predict(X_test_lstm, verbose=0)

transformer_mae = mean_absolute_error(y_test[:, 0], transformer_pred[:, 0])
transformer_rmse = np.sqrt(mean_squared_error(y_test[:, 0], transformer_pred[:, 0]))

logger.info(f"Transformer MAE: {transformer_mae:.4f}")
logger.info(f"Transformer RMSE: {transformer_rmse:.4f}")



# Code block 6
# Compile results
results = {
    'ARIMA': {
        'MAE': arima_mae, 
        'RMSE': arima_rmse,
        'Time': arima_time
    },
    'LSTM': {
        'MAE': lstm_mae, 
        'RMSE': lstm_rmse,
        'Time': lstm_time
    },
    'Transformer': {
        'MAE': transformer_mae, 
        'RMSE': transformer_rmse,
        'Time': transformer_time
    }
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# MAE comparison
mae_values = [results[m]['MAE'] for m in results.keys()]
axes[0].bar(results.keys(), mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
axes[0].set_title('Mean Absolute Error', fontweight='bold', fontsize=12)
axes[0].set_ylabel('MAE')
# RMSE comparison
rmse_values = [results[m]['RMSE'] for m in results.keys()]
axes[1].bar(results.keys(), rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
axes[1].set_title('Root Mean Squared Error', fontweight='bold', fontsize=12)
axes[1].set_ylabel('RMSE')
# Training time comparison
time_values = [results[m]['Time'] for m in results.keys()]
axes[2].bar(results.keys(), time_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
axes[2].set_title('Training Time', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Seconds')
plt.tight_layout()
plt.savefig('transformer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
logger.info("=== MODEL COMPARISON SUMMARY ===")
logger.info(f"{'Model':<15} {'MAE':<12} {'RMSE':<12} {'Time (s)':<12}")
for model, metrics in results.items():
    logger.info(f"{model:<15} {metrics['MAE']:<12.4f} {metrics['RMSE']:<12.4f} {metrics['Time']:<12.2f}")



# Code block 7
# Get test period dates
test_dates = ts_data.index[split_idx + seq_length:split_idx + seq_length + len(y_test)]

# Inverse transform predictions for visualization
lstm_pred_inv = scaler.inverse_transform(lstm_pred[:, 0].reshape(-1, 1)).flatten()
transformer_pred_inv = scaler.inverse_transform(transformer_pred[:, 0].reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten()

fig, ax = plt.subplots(figsize=(14, 6))

# Plot historical data
historical_dates = ts_data.index[:split_idx + seq_length]
ax.plot(historical_dates[-20:], ts_data.values[split_idx + seq_length - 20:split_idx + seq_length], 
        'k-', linewidth=2, label='Historical', alpha=0.7)

ax.plot(test_dates[:len(y_test_inv)], y_test_inv, 
        'o-', linewidth=2, markersize=8, label='Actual', color='black')

ax.plot(test_dates[:len(arima_forecast)], arima_forecast, 
        '--', linewidth=2, label='ARIMA', color='#1f77b4')
ax.plot(test_dates[:len(lstm_pred_inv)], lstm_pred_inv, 
        '-', linewidth=2, label='LSTM', color='#ff7f0e')
ax.plot(test_dates[:len(transformer_pred_inv)], transformer_pred_inv, 
        '-', linewidth=2, label='Transformer', color='#2ca02c')

# Add forecast boundary
ax.axvline(test_dates[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax.set_ylabel('Energy Consumption', fontsize=11)
ax.set_title('Energy Consumption Forecasts: ARIMA vs LSTM vs Transformer', 
             fontsize=13, fontweight='bold')
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('forecast_comparison.png', dpi=300, bbox_inches='tight')
plt.show()



# Code block 8
# Complete code for reproducibility
# All imports, data loading, model training, and evaluation
# See individual code blocks above for full implementation



logger.info("All images generated successfully!")
