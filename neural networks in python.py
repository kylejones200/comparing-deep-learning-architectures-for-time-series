
def main() -> None:
    # --- notebook cell (unparsed) ---
    # """Generated from Jupyter notebook: neural networks in python

    # Magics and shell lines are commented out. Run with a normal Python interpreter."""


    # # --- code cell ---

    # import torch
    # import torch.nn as nn
    # from torch.utils.data import DataLoader, TensorDataset
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import pandas as pd
    # from sklearn.metrics import mean_squared_error
    # from sklearn.preprocessing import MinMaxScaler
    #     Conv1D,
    #     Dense,
    #     Dropout,
    #     LayerNormalization,
    #     MultiHeadAttention,
    # )


    # # Function to prepare data for time series modeling
    # class _TransformerForecaster(nn.Module):
    #     """Transformer forecaster (auto-generated PyTorch replacement for Keras)."""
    #     def __init__(self, n_features: int, d_model: int = 256, nhead: int = 4,
    #                  ff_dim: int = 4, num_layers: int = 2,
    #                  output_size: int = 1, dropout: float = 0.0):
    #         super().__init__()
    #         self.proj = nn.Linear(n_features, d_model)
    #         layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True)
    #         self.encoder = nn.TransformerEncoder(layer, num_layers)
    #         self.fc = nn.Linear(d_model, output_size)

    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         x = self.proj(x)
    #         x = self.encoder(x)
    #         return self.fc(x[:, -1, :])

    # def _train_torch(model: nn.Module, X_train, y_train, *,
    #                  epochs: int = 50, batch_size: int = 32,
    #                  lr: float = 0.001, validation_split: float = 0.2,
    #                  patience: int = 15) -> nn.Module:
    #     """Standard training loop replacing  + model.fit()."""
    #     X_t = torch.FloatTensor(X_train)
    #     y_t = torch.FloatTensor(y_train)
    #     if y_t.dim() == 1:
    #         y_t = y_t.unsqueeze(1)
    #     n_val = max(1, int(len(X_t) * validation_split))
    #     X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    #     X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    #     loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     criterion = nn.MSELoss()
    #     best, wait = float("inf"), 0
    #     for _ in range(epochs):
    #         model.train()
    #         for xb, yb in loader:
    #             optimizer.zero_grad()
    #             criterion(model(xb), yb).backward()
    #             optimizer.step()
    #         model.eval()
    #         with torch.no_grad():
    #             val_loss = criterion(model(X_val), y_val).item()
    #         if val_loss < best:
    #             best, wait = val_loss, 0
    #         else:
    #             wait += 1
    #             if wait >= patience:
    #                 break
    #     return model


    # def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    #     """Replace model.predict()."""
    #     model.eval()
    #     with torch.no_grad():
    #         return model(torch.FloatTensor(X_test)).numpy()

    # def prepare_data(data, n_steps):
    #     X, y = [], []
    #     for i in range(len(data) - n_steps):
    #         X.append(data[i : i + n_steps])
    #         y.append(data[i + n_steps])
    #     return np.array(X), np.array(y)


    # # Function to load and preprocess data
    # def load_and_preprocess_data(filepath, n_steps=30):
    #     df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    #     df.sort_index(inplace=True)
    #     scaler = MinMaxScaler()
    #     scaled_data = scaler.fit_transform(df[["values"]])
    #     X, y = prepare_data(scaled_data, n_steps)
    #     train_size = int(len(X) * 0.8)
    #     X_train, X_test = X[:train_size], X[train_size:]
    #     y_train, y_test = y[:train_size], y[train_size:]
    #     return X_train, X_test, y_train, y_test, scaler


    # # Residual block for TCN
    # def residual_block(x, dilation_rate, filters):
    #     skip = x
    #     x = Conv1D(
    #         filters,
    #         kernel_size=3,
    #         dilation_rate=dilation_rate,
    #         padding="causal",
    #         activation="relu",
    #     )(x)
    #     x = Dropout(0.1)(x)
    #     x = Conv1D(
    #         filters,
    #         kernel_size=3,
    #         dilation_rate=dilation_rate,
    #         padding="causal",
    #         activation="relu",
    #     )(x)
    #     x = Dropout(0.4)(x)
    #     x = LayerNormalization()(x + skip)
    #     return x


    # # Transformer encoder block
    # def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    #     x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
    #         inputs, inputs
    #     )
    #     x = Dropout(dropout)(x)
    #     x = LayerNormalization(epsilon=1e-6)(x)
    #     res = x + inputs

    #     x = Dense(ff_dim, activation="relu")(res)
    #     x = Dropout(dropout)(x)
    #     x = Dense(inputs.shape[-1])(x)
    #     x = LayerNormalization(epsilon=1e-6)(x)
    #     return x + res


    # # Function to train and evaluate models
    # def run_models(X_train, X_test, y_train, y_test, n_steps):
    #     results = {}

    #     # FNN
    #     model_fnn = Sequential(
    #         [
    #             nn.Input(shape=(n_steps, 1)),
    #             nn.Flatten(),
    #             nn.Dense(64, activation="relu"),
    #             nn.Dense(32, activation="relu"),
    #             nn.Dense(1),
    #         ]
    #     )
    #         _train_torch(model_fnn, X_train, y_train)
    #     results["FNN"] = _predict_torch(model_fnn, X_test)

    #     # LSTM
    #     X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    #     X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    #     model_lstm = Sequential(
    #         [
    #             nn.Input(shape=(n_steps, 1)),
    #             nn.LSTM(50, activation="relu"),
    #             nn.Dense(1),
    #         ]
    #     )
    #         _train_torch(model_lstm, X_train_lstm, y_train)
    #     results["LSTM"] = _predict_torch(model_lstm, X_test_lstm)

    #     # CNN
    #     model_cnn = Sequential(
    #         [
    #             nn.Input(shape=(n_steps, 1)),
    #             nn.Conv1D(filters=64, kernel_size=3, activation="relu"),
    #             nn.MaxPooling1D(pool_size=2),
    #             nn.Flatten(),
    #             nn.Dense(50, activation="relu"),
    #             nn.Dense(1),
    #         ]
    #     )
    #         _train_torch(model_cnn, X_train_lstm, y_train)
    #     results["CNN"] = _predict_torch(model_cnn, X_test_lstm)

    #     # TCN
    #     inputs_tcn = nn.Input(shape=(n_steps, 1))
    #     x = inputs_tcn
    #     for i in range(4):
    #         x = residual_block(x, dilation_rate=2**i, filters=64)
    #     x = Dense(1)(x[:, -1, :])
    #     model_tcn = Model(inputs_tcn, x)
    #         _train_torch(model_tcn, X_train_lstm, y_train)
    #     results["TCN"] = _predict_torch(model_tcn, X_test_lstm)

    #     # Transformer
    #     inputs_transformer = Input(shape=(n_steps, 1))
    #     x = inputs_transformer
    #     for _ in range(4):
    #         x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
    #     x = Dense(1)(x[:, -1, :])
    #     model_transformer = Model(inputs_transformer, x)
    #         _train_torch(model_transformer, X_train_lstm, y_train)
    #     results["Transformer"] = _predict_torch(model_transformer, X_test_lstm)

    #     return results


    # # Function to evaluate and visualize results
    # def evaluate_and_plot(y_test, results, scaler):
    #     y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    #     predictions_inv = {
    #         name: scaler.inverse_transform(pred) for name, pred in results.items()
    #     }

    #     # Plot predictions
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(y_test_inv, label="True")
    #     for name, pred in predictions_inv.items():
    #         plt.plot(pred, label=f"Predicted ({name})")
    #     plt.title("Model Comparisons")
    #     plt.legend()
    #     plt.savefig("Model_Comparisons.png")
    #     plt.show()

    #     # Calculate and print MSE
    #     mse_scores = {
    #         name: mean_squared_error(y_test_inv, pred)
    #         for name, pred in predictions_inv.items()
    #     }
    #     for model, mse in mse_scores.items():
    #         print(f"{model} MSE: {mse:.3f}")
    #     return mse_scores



    # def main():
    #     # Main workflow
    #     filepath = "ercot_load_data.csv"  # Replace with your dataset file path
    #     n_steps = 30

    #     # Load and preprocess data
    #     X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(filepath, n_steps)

    #     # Train models and get predictions
    #     results = run_models(X_train, X_test, y_train, y_test, n_steps)

    #     # Evaluate and visualize results
    #     mse_scores = evaluate_and_plot(y_test, results, scaler)


    # if __name__ == "__main__":
    #     main()

if __name__ == "__main__":
    main()
