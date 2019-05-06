import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from models import *


def plot_series(timesteps, values, title):
    x = range(timesteps)
    y = values
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def prepare_data(data, window_size):
    batches = []
    labels = []
    for idx in range(len(data) - window_size - 1):
        batches.append(data[idx: idx + window_size])
        labels.append(data[idx + window_size])
    return np.array(batches), np.array(labels)


def simulation_mode(data, model, position_to_start_predicting, length_of_prediction):
    dataset_length = len(data)

    if dataset_length > position_to_start_predicting:
        prediction_data = np.zeros(data.shape)
        prediction_data[:position_to_start_predicting - 1] = data[:position_to_start_predicting - 1]
    else:
        prediction_data = np.zeros((dataset_length + length_of_prediction, 1))
        prediction_data[:dataset_length] = data

    for idx in range(position_to_start_predicting, position_to_start_predicting + length_of_prediction):
        input = prediction_data[idx - 50:idx]

        # Predict
        prediction = model.predict(input).round()

        # append prediction to prediction_data
        prediction_data[idx] = prediction

    return prediction_data


def plot_all_models():
    figure = plt.figure(1)

    y1 = series
    y2 = predictions_lstm[-length_of_prediction:]
    y3 = predictions_cnn[-length_of_prediction:]
    y4 = predictions_mlp[-length_of_prediction:]
    x = range(starting_point_of_prediction, starting_point_of_prediction + length_of_prediction)

    figure.add_subplot(2, 4, 1)
    plt.plot(range(len(series)), y1, label="Original Series", linestyle="solid", color='blue', linewidth=0.5)
    plt.title('Original Series')

    figure.add_subplot(2, 4, 2)
    plt.plot(x, y2, label="Simulated CNN", linestyle="solid", color='red')
    plt.title('Predicted LSTM')

    figure.add_subplot(2, 4, 3)
    plt.plot(x, y3, linestyle="solid", color='green')
    plt.title("Predicted CNN")

    figure.add_subplot(2, 4, 4)
    plt.plot(x, y4, linestyle="solid", color='black')
    plt.title("Predicted MLP")

    # Print losses. Only after training, not with loaded models
    if model_lstm.history:
        figure.add_subplot(2, 4, 6)
        plt.plot(model_lstm.history.history['loss'])
        plt.title("LSTM Loss"), plt.ylabel('loss'), plt.xlabel('epoch')

    if model_cnn.history:
        figure.add_subplot(2, 4, 7)
        plt.plot(model_cnn.history.history['loss'])
        plt.title("CNN Loss"), plt.ylabel('loss'), plt.xlabel('epoch')

    if model_mlp.history:
        figure.add_subplot(2, 4, 8)
        plt.plot(model_mlp.history.history['loss'])
        plt.title("MLP Loss"), plt.ylabel('loss'), plt.xlabel('epoch')

    plt.show()


# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])

# plot training set
# plot_series(1000, series, 'Original Data')

# define window size
window_size = 150

# define epochs
epochs = 300

# apply window size to construct a batches of training data and expected prediction in labels
batches, labels = prepare_data(series, window_size)

# # use as input all batches but the last one, to use as test
# batches = batches[:, :, 0]
# X = batches[:-1]
# y = labels[:-1]

# choose training data
# train_set = batches[:, :, 0]
train_set = batches
train_labels = labels

# initiate Models
model_mlp = MLPModel(window_size)
model_cnn = CNNModel(window_size)
model_lstm = LSTMModel(window_size)

# load models
# model_mlp.load_model('saved-models/mlp_1000.h5')
# model_cnn.load_model('saved-models/cnn_1000.h5')
# model_lstm.load_model('saved-models/lstm_1000.h5')

# print overview of models
# model_mlp.model.summary()
# model_cnn.model.summary()
# model_lstm.model.summary()

# train models
model_mlp.fit(train_set, train_labels, epochs, 2)
model_cnn.fit(train_set, train_labels, epochs, 2)
model_lstm.fit(train_set, train_labels, epochs, 0)

# save models
model_mlp.save_model('mlp_{0}.h5'.format(epochs))
model_cnn.save_model('cnn_{0}.h5'.format(epochs))
model_lstm.save_model('lstm_{0}.h5'.format(epochs))

# simulate next steps in the series and compare with original
starting_point_of_prediction = 1000
length_of_prediction = 400

# run simulation mode to predict the next values
predictions_mlp = simulation_mode(series, model_mlp, starting_point_of_prediction, length_of_prediction)
predictions_cnn = simulation_mode(series, model_cnn, starting_point_of_prediction, length_of_prediction)
predictions_lstm = simulation_mode(series, model_lstm, starting_point_of_prediction, length_of_prediction)

# plot both original series and simulated predictions
plot_all_models()
