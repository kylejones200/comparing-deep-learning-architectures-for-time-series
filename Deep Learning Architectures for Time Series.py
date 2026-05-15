"""Generated from Jupyter notebook: Deep Learning Architectures for Time Series

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
n = len(date_rng)
trend = np.linspace(0, 100, n)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.normal(0, 5, n)
y = trend + seasonality + noise

df = pd.DataFrame(data={"date": date_rng, "value": y})


## Feedforward Neural Networks: The Baseline
class _TransformerForecaster(nn.Module):
    """Transformer forecaster (auto-generated PyTorch replacement for Keras)."""
    def __init__(self, n_features: int, d_model: int = 256, nhead: int = 4,
                 ff_dim: int = 4, num_layers: int = 2,
                 output_size: int = 1, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 32,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 15) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["value"]])

# Prepare data for FNN
n_steps = 30
X, y = prepare_data(scaled_data, n_steps)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build FNN model

model_fnn = Sequential(
    [
        nn.Input(shape=(n_steps, 1)),
        nn.Flatten(),
        nn.Dense(64, activation="relu"),
        nn.Dense(32, activation="relu"),
        nn.Dense(1),
    ]
)

history_fnn = _train_torch(model_fnn, X_train, y_train)

# Evaluate FNN
y_pred_fnn = _predict_torch(model_fnn, X_test)
mse_fnn = mean_squared_error(y_test, y_pred_fnn)
print(f"FNN MSE: {mse_fnn}")
## Long Short-Term Memory (LSTM): Capturing Long-Range Dependencies
# Reshape input for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model_lstm = Sequential(
    [
        nn.Input(shape=(n_steps, 1)),
        nn.LSTM(50, activation="relu"),
        nn.Dense(1),
    ]
)


history_lstm = _train_torch(model_lstm, X_train_lstm, y_train)

# Evaluate LSTM
y_pred_lstm = _predict_torch(model_lstm, X_test_lstm)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
print(f"LSTM MSE: {mse_lstm}")
## Convolutional Neural Networks (CNN): Capturing Local Patterns
# Build CNN model
model_cnn = Sequential(
    [
        nn.Input(shape=(n_steps, 1)),
        nn.Conv1D(filters=64, kernel_size=3, activation="relu"),
        nn.MaxPooling1D(pool_size=2),
        nn.Flatten(),
        nn.Dense(50, activation="relu"),
        nn.Dense(1),
    ]
)

history_cnn = _train_torch(model_cnn, X_train_lstm, y_train)

# Evaluate CNN
y_pred_cnn = _predict_torch(model_cnn, X_test_lstm)
mse_cnn = mean_squared_error(y_test, y_pred_cnn)
print(f"CNN MSE: {mse_cnn}")
## Temporal Convolutional Networks (TCN): Combining CNN and RNN Strengths


def residual_block(x, dilation_rate, filters):
    skip = x
    x = Conv1D(
        filters,
        kernel_size=3,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
    )(x)
    x = Dropout(0.1)(x)
    x = Conv1D(
        filters,
        kernel_size=3,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
    )(x)
    x = Dropout(0.4)(x)
    x = LayerNormalization()(x + skip)
    return x


# Build TCN model
inputs = nn.Input(shape=(n_steps, 1))
x = inputs
for i in range(4):
    x = residual_block(x, dilation_rate=2**i, filters=64)
x = Dense(1)(x[:, -1, :])
model_tcn = Model(inputs, x)

history_tcn = _train_torch(model_tcn, X_train_lstm, y_train)

# Evaluate TCN
y_pred_tcn = _predict_torch(model_tcn, X_test_lstm)
mse_tcn = mean_squared_error(y_test, y_pred_tcn)
print(f"TCN MSE: {mse_tcn}")
## Transformer: Attention-based Sequence Modeling


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        inputs, inputs
    )
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


# Build Transformer model
inputs = Input(shape=(n_steps, 1))
x = inputs
for _ in range(4):
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
x = Dense(1)(x[:, -1, :])
model_transformer = Model(inputs, x)

history_transformer = _train_torch(model_transformer, X_train_lstm, y_train)

# Evaluate Transformer
y_pred_transformer = _predict_torch(model_transformer, X_test_lstm)
mse_transformer = mean_squared_error(y_test, y_pred_transformer)
print(f"Transformer MSE: {mse_transformer}")


## Comparative Analysis
def plot_predictions(true, predictions, title):
    plt.figure(figsize=(12, 6))
    plt.plot(true, label="True")
    for name, pred in predictions.items():
        plt.plot(pred, label=f"Predicted ({name})")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.show()


# Inverse transform predictions
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_fnn_inv = scaler.inverse_transform(y_pred_fnn)
y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
y_pred_cnn_inv = scaler.inverse_transform(y_pred_cnn)
y_pred_tcn_inv = scaler.inverse_transform(y_pred_tcn)
y_pred_transformer_inv = scaler.inverse_transform(y_pred_transformer)

# Plot predictions
predictions = {
    "FNN": y_pred_fnn_inv,
    "LSTM": y_pred_lstm_inv,
    "CNN": y_pred_cnn_inv,
    "TCN": y_pred_tcn_inv,
    "Transformer": y_pred_transformer_inv,
}
plot_predictions(y_test_inv, predictions, "Model Comparisons")

# Compare MSE
mse_scores = {
    "FNN": mse_fnn,
    "LSTM": mse_lstm,
    "CNN": mse_cnn,
    "TCN": mse_tcn,
    "Transformer": mse_transformer,
}
for model, mse in mse_scores.items():
    print(f"{model} MSE: {mse:.3f}")


# --- code cell ---

## Transformer: Attention-based Sequence Modeling


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        inputs, inputs
    )
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


# Build Transformer model
inputs = Input(shape=(n_steps, 1))
x = inputs
for _ in range(4):
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
x = Dense(1)(x[:, -1, :])
model_transformer = Model(inputs, x)

history_transformer = _train_torch(model_transformer, X_train_lstm, y_train)

# Evaluate Transformer
y_pred_transformer = _predict_torch(model_transformer, X_test_lstm)
mse_transformer = mean_squared_error(y_test, y_pred_transformer)
print(f"Transformer MSE: {mse_transformer}")


# --- code cell ---

# Compare MSE
mse_scores = {
    "FNN": mse_fnn,
    "LSTM": mse_lstm,
    "CNN": mse_cnn,
    "TCN": mse_tcn,
    "Transformer": mse_transformer,
}
for model, mse in mse_scores.items():
    print(f"{model} MSE: {mse:.4f}")


# --- code cell ---

# Convert predictions and true values to DataFrame for easier resampling
df_results = pd.DataFrame(
    {
        "date": df["date"][-len(y_test_inv) :],  # Match the test set dates
        "true": y_test_inv.flatten(),
        "FNN": y_pred_fnn_inv.flatten(),
        "LSTM": y_pred_lstm_inv.flatten(),
        "CNN": y_pred_cnn_inv.flatten(),
        "TCN": y_pred_tcn_inv.flatten(),
        "Transformer": y_pred_transformer_inv.flatten(),
    }
)

# Set the 'date' column as the index
df_results.set_index("date", inplace=True)

# Resample to a weekly average
df_resampled = df_results.resample("W").mean()


# Plot the resampled predictions
def plot_resampled_predictions(resampled_df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(resampled_df["true"], label="True", linewidth=2)
    for col in resampled_df.columns:
        if col != "true":
            plt.plot(resampled_df[col], label=f"Predicted ({col})", alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.show()


plot_resampled_predictions(df_resampled, "Model Comparisons (Weekly Resampled)")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("ercot_load_data.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df.sort_index(inplace=True)
df = df[df["values"] >= 60]  # Filter out invalid values
values = df["values"]
dates = df.index

# Resample the data to hourly for modeling
df_resampled = df.resample("H").mean()

# Normalize the data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df_resampled[["values"]])


# Prepare data for time series modeling
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


# Set parameters
n_steps = 48  # For 48 hours (2 days) of history
X, y = prepare_data(scaled_values, n_steps)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM and CNN models
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# Define a helper function to plot predictions
def plot_predictions(true, predictions, title, resample_freq="D"):
    # Resample to smooth out noise
    df_results = pd.DataFrame({"date": dates[-len(true) :], "true": true.flatten()})
    for name, pred in predictions.items():
        df_results[name] = pred.flatten()
    df_results.set_index("date", inplace=True)
    df_resampled = df_results.resample(resample_freq).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled["true"], label="True", linewidth=2)
    for name in predictions.keys():
        plt.plot(df_resampled[name], label=f"Predicted ({name})", alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.savefig("Plot.png")
    plt.show()


# Build and train all models
models = {
    "FNN": Sequential(
        [
            nn.Flatten(input_shape=(n_steps, 1)),
            nn.Dense(64, activation="relu"),
            nn.Dense(32, activation="relu"),
            nn.Dense(1),
        ]
    ),
    "LSTM": Sequential(
        [
            nn.LSTM(50, activation="relu", input_shape=(n_steps, 1)),
            nn.Dense(1),
        ]
    ),
    "CNN": Sequential(
        [
            nn.Conv1D(
                filters=64, kernel_size=3, activation="relu", input_shape=(n_steps, 1)
            ),
            nn.MaxPooling1D(pool_size=2),
            nn.Flatten(),
            nn.Dense(50, activation="relu"),
            nn.Dense(1),
        ]
    ),
}

# Add TCN model


def build_tcn(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(4):
        x = residual_block(x, dilation_rate=2**i, filters=64)
    x = Dense(1)(x[:, -1, :])
    return Model(inputs, x)


def residual_block(x, dilation_rate, filters):
    skip = x
    x = Conv1D(
        filters,
        kernel_size=3,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
    )(x)
    x = Dropout(0.1)(x)
    x = Conv1D(
        filters,
        kernel_size=3,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
    )(x)
    x = Dropout(0.1)(x)
    x = LayerNormalization()(x + skip)
    return x


models["TCN"] = build_tcn((n_steps, 1))

# Compile all models
for name, model in models.items():
    
# Train models
histories = {}
for name, model in models.items():
    print(f"Training {name}...")
    history = _train_torch(model, X_train_lstm, y_train)
    histories[name] = history

# Evaluate models
predictions = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    y_pred = _predict_torch(model, X_test_lstm)
    predictions[name] = scaler.inverse_transform(y_pred)

# Plot predictions
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
plot_predictions(y_test_inv, predictions, "Model Comparisons (Hourly Resampled)")


# --- code cell ---


# Prepare resampled data and ensure proper alignment
def plot_predictions(true, predictions, title, resample_freq="D"):
    # Create DataFrame for plotting
    df_results = pd.DataFrame({"date": dates[-len(true) :], "true": true.flatten()})
    for name, pred in predictions.items():
        df_results[name] = pred.flatten()

    # Debug: Print the initial df_results
    print("Initial df_results:\n", df_results.head())

    # Set the 'date' column as the index and resample
    df_results.set_index("date", inplace=True)
    df_resampled = df_results.resample(resample_freq).mean()

    # Debug: Print the resampled data
    print("Resampled Data:\n", df_resampled.head())

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled["true"], label="True", linewidth=2)
    for name in predictions.keys():
        plt.plot(df_resampled[name], label=f"Predicted ({name})", alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.grid()
    plt.show()


# Evaluate models and plot predictions
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
plot_predictions(
    y_test_inv, predictions, "Model Comparisons (Resampled to Daily)", resample_freq="D"
)
