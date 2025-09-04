"""
LSTM Stock Price Prediction Module
-----------------------------------
This script provides helper functions for:
- Loading and preprocessing stock price data
- Building and training an LSTM model
- Making predictions (single step and sequence-based)
- Visualizing results

Updated to Python 3 and modern Keras/TensorFlow APIs.
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ===================== Data Preprocessing =====================
def normalise_windows(window_data):
    """
    Normalize each sequence by dividing by the first value and subtracting 1.
    This makes the data scale-invariant.
    """
    normalised_data = []
    for window in window_data:
        normalised_window = [(float(p) / float(window[0]) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def load_data(filename, seq_len=50, normalise_window=True):
    """
    Load stock price data from a text/CSV file and create training/test sets.

    Args:
        filename (str): Path to dataset (1D stock prices).
        seq_len (int): Sequence length (time window).
        normalise_window (bool): Whether to normalize each sequence.

    Returns:
        (x_train, y_train, x_test, y_test): Training and testing sets.
    """
    with open(filename, 'r') as f:
        data = f.read().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result, dtype=float)

    train_size = int(round(0.9 * result.shape[0]))
    train = result[:train_size]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[train_size:, :-1]
    y_test = result[train_size:, -1]

    # Reshape for LSTM [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test


# ===================== Model Building =====================
def build_model(layers):
    """
    Build and compile an LSTM model.

    Args:
        layers (list): [input_dim, hidden_units_1, hidden_units_2, output_dim]

    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = Sequential()

    model.add(LSTM(units=layers[1],
                   input_shape=(None, layers[0]),
                   return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : {:.2f} sec".format(time.time() - start))
    return model


# ===================== Prediction Functions =====================
def predict_point_by_point(model, data):
    """
    Predict each timestep given the last sequence of true data.
    (1 step ahead prediction each time)
    """
    predicted = model.predict(data)
    return np.reshape(predicted, (predicted.size,))


def predict_sequence_full(model, data, window_size):
    """
    Predict a full sequence by feeding back predictions.
    """
    curr_frame = data[0]
    predicted = []
    for _ in range(len(data)):
        pred = model.predict(curr_frame[np.newaxis, :, :])[0, 0]
        predicted.append(pred)
        curr_frame = np.vstack((curr_frame[1:], [pred]))
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len=50):
    """
    Predict multiple sequences of length `prediction_len`.
    """
    prediction_seqs = []
    for i in range(len(data) // prediction_len):
        curr_frame = data[i * prediction_len]
        predicted = []
        for _ in range(prediction_len):
            pred = model.predict(curr_frame[np.newaxis, :, :])[0, 0]
            predicted.append(pred)
            curr_frame = np.vstack((curr_frame[1:], [pred]))
        prediction_seqs.append(predicted)
    return prediction_seqs


# ===================== Visualization & Metrics =====================
def plot_results(true_data, predicted_data, title="Stock Price Prediction"):
    """
    Plot true vs predicted stock prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true_data, label="True Data")
    plt.plot(predicted_data, label="Prediction")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.show()


def evaluate_model(y_true, y_pred):
    """
    Evaluate predictions using MAE, RMSE, and R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
