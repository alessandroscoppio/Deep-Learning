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
        prediction_data = np.zeros((dataset_length+length_of_prediction, 1))
        prediction_data[:dataset_length] = data

    for idx in range(position_to_start_predicting, position_to_start_predicting + length_of_prediction):
        input = prediction_data[idx-50:idx]

        # Predict
        prediction = model.predict(input).round()

        # append prediction to prediction_data
        prediction_data[idx] = prediction

    return prediction_data


# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])
# plot training set
# plot_series(1000, series, 'Original Data')

# define window size
window_size = 50

# define epochs
epochs = 200

# apply window size to construct a batches of training data and expected prediction in labels
batches, labels = prepare_data(series, window_size)

# # use as input all batches but the last one, to use as test
# batches = batches[:, :, 0]
# X = batches[:-1]
# y = labels[:-1]

# # Multi-Layer Perceptron Model
# MLP_model = MLPModel(window_size)
#
# # train
# MLP_model.fit(X, y, epochs)
#
# # # predict
# prediction = MLP_model.predict(batches[-1])
# print("Predicted: {0}\nExpecred:  {1}".format(prediction, labels[-1]))

# Choose training data
train_set = batches
train_labels = labels

# Initiate Models
model_mlp = MLPModel(window_size)
model_mlp.fit(train_set, train_labels, epochs)
model_cnn = CNNModel(window_size)
model_cnn.fit(train_set, train_labels, epochs)


# simulate next steps in the series and compare with original
starting_point_of_prediction = 1000
length_of_prediction = 200

# Run simulation mode to predict the next values
predictions_cnn = simulation_mode(series, model_cnn, starting_point_of_prediction, length_of_prediction)
predictions_mlp = simulation_mode(series, model_mlp, starting_point_of_prediction, length_of_prediction)

# Plot both original series and simulated predictions
y1 = series
y2 = predictions_cnn
y3 = predictions_mlp
plt.plot(range(len(series)), y1, label="Original Series", linestyle="dotted", color='blue')
plt.plot(range(len(predictions_cnn)), y2, label="Simulated CNN", linestyle="dotted", color='red')
plt.plot(range(len(predictions_mlp)), y3, label="Simulated MLP", linestyle="dotted", color='green')
plt.title("Predicted Values")
plt.legend()
plt.show()
