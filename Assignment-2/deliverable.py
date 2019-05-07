import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model


class MLPModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.history = None
        self.model = Sequential()
        self.model.add(Dense(100, activation='relu', input_dim=self.input_size))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0):
        # fit model
        X = X.reshape((X.shape[0], X.shape[1]))
        self.history = self.model.fit(X, y, batch_size=10, epochs=epochs, verbose=verbose)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)


class CNNModel:
    def __init__(self, input_size):
        self.input_size = input_size
        self.history = None
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', batch_input_shape=(None, self.input_size, 1)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0, batch_size=32):
        self.history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, x):
        x_input = x.reshape((1, self.input_size, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction[0]

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)


class LSTMModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.history = None
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.input_size, 1)))
        # self.model.add(Dropout(0.5))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0, batch_size=32):
        # fit model
        self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)


class CNNLSTMModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.history = None
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 25, 1)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0):
        # fit model
        X = X.reshape((X.shape[0], 2, 25, 1))
        self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, 2, 25, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)


def prepare_data(data, window_size):
    batches = []
    labels = []
    for idx in range(len(data) - window_size - 1):
        batches.append(data[idx: idx + window_size])
        labels.append(data[idx + window_size])
    return np.array(batches), np.array(labels)


def simulation_mode(data, model, window_size, position_to_start_predicting, length_of_prediction):
    dataset_length = len(data)

    if dataset_length > position_to_start_predicting:
        prediction_data = np.zeros(data.shape)
        prediction_data[:position_to_start_predicting - 1] = data[:position_to_start_predicting - 1]
    else:
        prediction_data = np.zeros((dataset_length + length_of_prediction, 1))
        prediction_data[:dataset_length] = data

    for idx in range(position_to_start_predicting, position_to_start_predicting + length_of_prediction):
        input = prediction_data[idx - window_size:idx]

        # Predict
        prediction = model.predict(input)

        # append prediction to prediction_data
        prediction_data[idx] = prediction

    return prediction_data


def plot_multiple_models(models, predictions, titles, rows, columns):
    figure = plt.figure(1)
    for idx in range(len(models)):
        figure.add_subplot(rows, columns, idx + 1)
        plt.plot(predictions[idx], linewidth=0.5, linestyle="solid", color='blue')
        plt.title(titles[idx])

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.show()


def experiment_with_batch_and_window_size():
    # Experiment with these values and their combinations
    window_sizes = [10, 50, 100, 450]
    batch_sizes = [1, 32, 256]

    # Store all variables in lists
    models = []
    predictions = []
    titles = []

    network_all = len(window_sizes) * len(batch_sizes)
    network_count = 1

    # Iterate over all combinations
    for win_size in window_sizes:
        training_set, training_labels = prepare_data(series, win_size)
        for bat_size in batch_sizes:
            title = "Window size of {0} and batch size of {1}".format(win_size, bat_size)
            print("Training {0}/{1}: {2}".format(network_count, network_all, title))
            model = CNNModel(win_size)
            model.fit(training_set, training_labels, epochs=300, verbose=0, batch_size=bat_size)
            prediction = simulation_mode(
                data=series,
                model=model,
                window_size=win_size,
                position_to_start_predicting=1000,
                length_of_prediction=200
            )
            models.append(model)
            predictions.append(prediction)
            titles.append(title)

            network_count += 1

    plot_multiple_models(models, predictions, titles, 4, 3)


def experiment_normalization():
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(series)
    series_n = scaler.transform(series)

    batches_n, labels_n = prepare_data(series_n, window_size)
    batches, labels = prepare_data(series, window_size)

    model_cnn_n = CNNModel(window_size)
    model_cnn_n.fit(batches_n, labels_n, epochs, 2)
    model_cnn = CNNModel(window_size)
    model_cnn.fit(batches, labels, epochs, 2)

    model_mlp_n = MLPModel(window_size)
    model_mlp_n.fit(batches_n, labels_n, epochs, 2)
    model_mlp = MLPModel(window_size)
    model_mlp.fit(batches, labels, epochs, 2)

    model_lstm_n = LSTMModel(window_size)
    model_lstm_n.fit(batches_n, labels_n, epochs, 2)
    model_lstm = LSTMModel(window_size)
    model_lstm.fit(batches, labels, epochs, 2)

    predictions_cnn_n = scaler.inverse_transform(
        simulation_mode(series_n, model_cnn_n, window_size, starting_point_of_prediction, length_of_prediction))
    predictions_cnn = simulation_mode(series, model_cnn, window_size, starting_point_of_prediction,
                                      length_of_prediction)

    predictions_mlp_n = scaler.inverse_transform(
        simulation_mode(series_n, model_mlp_n, window_size, starting_point_of_prediction, length_of_prediction))
    predictions_mlp = simulation_mode(series, model_mlp, window_size, starting_point_of_prediction,
                                      length_of_prediction)

    predictions_lstm_n = scaler.inverse_transform(
        simulation_mode(series_n, model_lstm_n, window_size, starting_point_of_prediction, length_of_prediction))
    predictions_lstm = simulation_mode(series, model_lstm, window_size, starting_point_of_prediction,
                                       length_of_prediction)

    plot_multiple_models(
        [model_cnn_n, model_mlp_n, model_lstm_n, model_cnn, model_mlp, model_lstm],
        [predictions_cnn_n, predictions_mlp_n, predictions_lstm_n, predictions_cnn, predictions_mlp, predictions_lstm],
        ["CNN with data normalization", "MLP with data normalization", "LSTM with data normalization",
         "CNN with original data", "MLP with original data", "LSTM with original data"],
        2,
        3
    )


def run_model(model):
    # apply window size to construct a batches of training data and expected prediction in labels
    batches, labels = prepare_data(series, window_size)

    # Load model from memory (if already trained once)
    # model.load_model('saved-models/cnn_1000.h5')

    # Train model
    model.fit(batches, labels, epochs, 2)

    # Save model
    model.save_model('cnn_{0}.h5'.format(epochs))

    # Run simulation model and retrieve predictions
    predictions = simulation_mode(series, model, window_size, starting_point_of_prediction, length_of_prediction)

    plt.plot(predictions, linewidth=0.5, linestyle="solid", color='blue')
    plt.show()


# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])

# define window size
window_size = 50

# define epochs
epochs = 200

# initiate Models
model_mlp = MLPModel(window_size)
model_cnn = CNNModel(window_size)
model_lstm = LSTMModel(window_size)
model_cnnlstm = CNNLSTMModel(window_size)

# simulate next steps in the series and compare with original
starting_point_of_prediction = 1000
length_of_prediction = 200

# Run any of the models
run_model(model_cnn)  # Change to any of the other models from above

# Run experiments. Uncomment to execute
# experiment_with_batch_and_window_size()
# experiment_normalization()
