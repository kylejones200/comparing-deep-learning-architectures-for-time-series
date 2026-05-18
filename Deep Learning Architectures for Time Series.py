"""Generated from Jupyter notebook: Deep Learning Architectures for Time Series

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
)


def build_tcn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i in range(4):
        x = residual_block(x, dilation_rate=2**i, filters=64)
    x = Dense(1)(x[:, -1, :])
    return tf.keras.Model(inputs, x)


def plot_predictions(true, predictions, title, resample_freq="D"):
    df_results = pd.DataFrame({"date": dates[-len(true) :], "true": true.flatten()})
    for name, pred in predictions.items():
        df_results[name] = pred.flatten()
    print("Initial df_results:\n", df_results.head())
    df_results.set_index("date", inplace=True)
    df_resampled = df_results.resample(resample_freq).mean()
    print("Resampled Data:\n", df_resampled.head())
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


def plot_resampled_predictions(resampled_df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(resampled_df["true"], label="True", linewidth=2)
    for col in resampled_df.columns:
        if col != "true":
            plt.plot(resampled_df[col], label=f"Predicted ({col})", alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.show()


def prepare_data(data, n_steps):
    X, y = ([], [])
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps])
    return (np.array(X), np.array(y))


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


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        inputs, inputs
    )
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-06)(x)
    res = x + inputs
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-06)(x)
    return x + res


def generate_synthetic_time_series_data() -> None:
    np.random.seed(42)
    date_rng = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
    n = len(date_rng)
    trend = np.linspace(0, 100, n)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 5, n)
    y = trend + seasonality + noise
    df = pd.DataFrame(data={"date": date_rng, "value": y})
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["value"]])
    n_steps = 30
    X, y = prepare_data(scaled_data, n_steps)
    train_size = int(len(X) * 0.8)
    X_train, X_test = (X[:train_size], X[train_size:])
    y_train, y_test = (y[:train_size], y[train_size:])
    model_fnn = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_steps, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model_fnn.compile(optimizer="adam", loss="mse")
    model_fnn.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0
    )
    y_pred_fnn = model_fnn.predict(X_test)
    mse_fnn = mean_squared_error(y_test, y_pred_fnn)
    print(f"FNN MSE: {mse_fnn}")
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model_lstm = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_steps, 1)),
            tf.keras.layers.LSTM(50, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model_lstm.compile(optimizer="adam", loss="mse")
    model_lstm.fit(
        X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0
    )
    y_pred_lstm = model_lstm.predict(X_test_lstm)
    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    print(f"LSTM MSE: {mse_lstm}")
    model_cnn = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_steps, 1)),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model_cnn.compile(optimizer="adam", loss="mse")
    model_cnn.fit(
        X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0
    )
    y_pred_cnn = model_cnn.predict(X_test_lstm)
    mse_cnn = mean_squared_error(y_test, y_pred_cnn)
    print(f"CNN MSE: {mse_cnn}")
    inputs = tf.keras.layers.Input(shape=(n_steps, 1))
    x = inputs
    for i in range(4):
        x = residual_block(x, dilation_rate=2**i, filters=64)

    x = Dense(1)(x[:, -1, :])
    model_tcn = tf.keras.Model(inputs, x)
    model_tcn.compile(optimizer="adam", loss="mse")
    model_tcn.fit(
        X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0
    )
    y_pred_tcn = model_tcn.predict(X_test_lstm)
    mse_tcn = mean_squared_error(y_test, y_pred_tcn)
    print(f"TCN MSE: {mse_tcn}")
    inputs = tf.keras.Input(shape=(n_steps, 1))
    x = inputs
    for _ in range(4):
        x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)

    x = Dense(1)(x[:, -1, :])
    model_transformer = tf.keras.Model(inputs, x)
    model_transformer.compile(optimizer="adam", loss="mse")
    model_transformer.fit(
        X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0
    )
    y_pred_transformer = model_transformer.predict(X_test_lstm)
    mse_transformer = mean_squared_error(y_test, y_pred_transformer)
    print(f"Transformer MSE: {mse_transformer}")
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_fnn_inv = scaler.inverse_transform(y_pred_fnn)
    y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
    y_pred_cnn_inv = scaler.inverse_transform(y_pred_cnn)
    y_pred_tcn_inv = scaler.inverse_transform(y_pred_tcn)
    y_pred_transformer_inv = scaler.inverse_transform(y_pred_transformer)
    predictions = {
        "FNN": y_pred_fnn_inv,
        "LSTM": y_pred_lstm_inv,
        "CNN": y_pred_cnn_inv,
        "TCN": y_pred_tcn_inv,
        "Transformer": y_pred_transformer_inv,
    }
    plot_predictions(y_test_inv, predictions, "Model Comparisons")
    mse_scores = {
        "FNN": mse_fnn,
        "LSTM": mse_lstm,
        "CNN": mse_cnn,
        "TCN": mse_tcn,
        "Transformer": mse_transformer,
    }
    for model, mse in mse_scores.items():
        print(f"{model} MSE: {mse:.3f}")


def transformer_attention_based_sequence_modeling() -> None:
    inputs = tf.keras.Input(shape=(n_steps, 1))
    x = inputs
    for _ in range(4):
        x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)

    x = Dense(1)(x[:, -1, :])
    model_transformer = tf.keras.Model(inputs, x)
    model_transformer.compile(optimizer="adam", loss="mse")
    model_transformer.fit(
        X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0
    )
    y_pred_transformer = model_transformer.predict(X_test_lstm)
    mse_transformer = mean_squared_error(y_test, y_pred_transformer)
    print(f"Transformer MSE: {mse_transformer}")


def compare_mse() -> None:
    mse_scores = {
        "FNN": mse_fnn,
        "LSTM": mse_lstm,
        "CNN": mse_cnn,
        "TCN": mse_tcn,
        "Transformer": mse_transformer,
    }
    for model, mse in mse_scores.items():
        print(f"{model} MSE: {mse:.4f}")


def convert_predictions_and_true_values_to_dataframe() -> None:
    df_results = pd.DataFrame(
        {
            "date": df["date"][-len(y_test_inv) :],
            "true": y_test_inv.flatten(),
            "FNN": y_pred_fnn_inv.flatten(),
            "LSTM": y_pred_lstm_inv.flatten(),
            "CNN": y_pred_cnn_inv.flatten(),
            "TCN": y_pred_tcn_inv.flatten(),
            "Transformer": y_pred_transformer_inv.flatten(),
        }
    )
    df_results.set_index("date", inplace=True)
    df_resampled = df_results.resample("W").mean()
    plot_resampled_predictions(df_resampled, "Model Comparisons (Weekly Resampled)")


def load_data() -> None:
    df = pd.read_csv("ercot_load_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df = df[df["values"] >= 60]
    df["values"]
    df_resampled = df.resample("H").mean()
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_resampled[["values"]])
    n_steps = 48
    X, y = prepare_data(scaled_values, n_steps)
    train_size = int(len(X) * 0.8)
    X_train, X_test = (X[:train_size], X[train_size:])
    y_train, y_test = (y[:train_size], y[train_size:])
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    models = {
        "FNN": tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(n_steps, 1)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        ),
        "LSTM": tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(50, activation="relu", input_shape=(n_steps, 1)),
                tf.keras.layers.Dense(1),
            ]
        ),
        "CNN": tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    activation="relu",
                    input_shape=(n_steps, 1),
                ),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        ),
    }
    models["TCN"] = build_tcn((n_steps, 1))
    for name, model in models.items():
        model.compile(optimizer="adam", loss="mse")

    histories = {}
    for name, model in models.items():
        print(f"Training {name}...")
        history = model.fit(
            X_train_lstm,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )
        histories[name] = history

    predictions = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_lstm)
        predictions[name] = scaler.inverse_transform(y_pred)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    plot_predictions(y_test_inv, predictions, "Model Comparisons (Hourly Resampled)")


def prepare_resampled_data_and_ensure_proper_alignme() -> None:
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    plot_predictions(
        y_test_inv,
        predictions,
        "Model Comparisons (Resampled to Daily)",
        resample_freq="D",
    )


def main() -> None:
    generate_synthetic_time_series_data()
    transformer_attention_based_sequence_modeling()
    compare_mse()
    convert_predictions_and_true_values_to_dataframe()
    load_data()
    prepare_resampled_data_and_ensure_proper_alignme()


if __name__ == "__main__":
    main()
