"""Generated from Jupyter notebook: Tensorflow

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# !pip install tensorflow  # Jupyter-only


# --- code cell ---

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic data
np.random.seed(42)
time = np.arange(100)
data = 10 + 0.5 * time + np.sin(0.2 * time) + np.random.normal(scale=1.0, size=100)


# Create lagged features
def create_features(data, lag=3):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i : i + lag])
        y.append(data[i + lag])
    return np.array(X), np.array(y)



def main():
    lag = 3
    X, y = create_features(data, lag=lag)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a simple feedforward neural network
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(lag,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

    # Evaluate and predict
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred, label="Predicted", color="Red")
    plt.title(f"Feedforward Neural Network Forecast \n MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("NN_forecast.png")
    plt.show()


    # --- code cell ---

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred, label="Predicted", color="Red")
    plt.title("Feedforward Neural Network Forecast")
    plt.legend()
    plt.savefig("NN_forecast.png")
    plt.show()


    # --- code cell ---

    from tensorflow.keras.layers import SimpleRNN

    # Build an RNN model
    model = tf.keras.Sequential(
        [SimpleRNN(50, activation="relu", input_shape=(lag, 1)), tf.keras.layers.Dense(1)]
    )

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Reshape input for RNN (samples, timesteps, features)
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train the model
    model.fit(
        X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1
    )

    # Predict
    y_pred_rnn = model.predict(X_test_rnn)
    mape = mean_absolute_percentage_error(y_test, y_pred_rnn)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred_rnn, label="Predicted", color="Red")
    plt.title(f"Recurrent Neural Network Forecast \n MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("RNN_forecast.png")
    plt.show()


    # --- code cell ---

    from tensorflow.keras.layers import LSTM

    # Build an LSTM model
    model = tf.keras.Sequential(
        [LSTM(50, activation="relu", input_shape=(lag, 1)), tf.keras.layers.Dense(1)]
    )

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Train the model
    model.fit(
        X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1
    )

    # Predict
    y_pred_lstm = model.predict(X_test_rnn)
    mape = mean_absolute_percentage_error(y_test, y_pred_lstm)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred_lstm, label="Predicted", color="Red")
    plt.title(f"LSTM Forecast. MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("LSTM_forecast.png")
    plt.show()


    # --- code cell ---

    from sklearn.metrics import mean_absolute_percentage_error

    mean_absolute_percentage_error(y, y_pred)


if __name__ == "__main__":
    main()
