"""Generated from Jupyter notebook: Tensorflow

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, SimpleRNN


def create_features(data, lag=3):
    X, y = ([], [])
    for i in range(len(data) - lag):
        X.append(data[i : i + lag])
        y.append(data[i + lag])
    return (np.array(X), np.array(y))


def generate_synthetic_data() -> None:
    np.random.seed(42)

    time = np.arange(100)

    data = 10 + 0.5 * time + np.sin(0.2 * time) + np.random.normal(scale=1.0, size=100)

    lag = 3

    X, y = create_features(data, lag=lag)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(lag,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    model.summary()

    model.fit(
        X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1
    )

    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    plt.figure(figsize=(10, 6))

    plt.plot(y_test, label="Actual", color="Blue")

    plt.plot(y_pred, label="Predicted", color="Red")

    plt.title(f"Feedforward Neural Network Forecast \n MAPE: {mape:.3f}")

    plt.legend()

    plt.savefig("NN_forecast.png")

    plt.show()


def plot_results() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(y_test, label="Actual", color="Blue")

    plt.plot(y_pred, label="Predicted", color="Red")

    plt.title("Feedforward Neural Network Forecast")

    plt.legend()

    plt.savefig("NN_forecast.png")

    plt.show()


def build_an_rnn_model() -> None:
    model = tf.keras.Sequential(
        [
            SimpleRNN(50, activation="relu", input_shape=(lag, 1)),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    model.summary()

    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model.fit(
        X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1
    )

    y_pred_rnn = model.predict(X_test_rnn)

    mape = mean_absolute_percentage_error(y_test, y_pred_rnn)

    plt.figure(figsize=(10, 6))

    plt.plot(y_test, label="Actual", color="Blue")

    plt.plot(y_pred_rnn, label="Predicted", color="Red")

    plt.title(f"Recurrent Neural Network Forecast \n MAPE: {mape:.3f}")

    plt.legend()

    plt.savefig("RNN_forecast.png")

    plt.show()


def build_an_lstm_model() -> None:
    model = tf.keras.Sequential(
        [LSTM(50, activation="relu", input_shape=(lag, 1)), tf.keras.layers.Dense(1)]
    )

    model.compile(optimizer="adam", loss="mse")

    model.summary()

    model.fit(
        X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1
    )

    y_pred_lstm = model.predict(X_test_rnn)

    mape = mean_absolute_percentage_error(y_test, y_pred_lstm)

    plt.figure(figsize=(10, 6))

    plt.plot(y_test, label="Actual", color="Blue")

    plt.plot(y_pred_lstm, label="Predicted", color="Red")

    plt.title(f"LSTM Forecast. MAPE: {mape:.3f}")

    plt.legend()

    plt.savefig("LSTM_forecast.png")

    plt.show()


def notebook_step_006() -> None:
    mean_absolute_percentage_error(y, y_pred)


def main() -> None:
    generate_synthetic_data()
    plot_results()
    build_an_rnn_model()
    build_an_lstm_model()
    notebook_step_006()


if __name__ == "__main__":
    main()
