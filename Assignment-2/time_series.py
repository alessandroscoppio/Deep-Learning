# univariate mlp example
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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


def simulation_mode(original_input, model, steps):
    input = original_input
    predictions = []
    for step in range(steps):
        prediction = model.predict(input)
        predictions.append(prediction)
        new_input = np.zeros(len(input))
        new_input[:-1] = input[1:]
        new_input[-1] = prediction
        input = new_input

    return np.array(predictions)


# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])
# plot training set
plot_series(1000, series, 'Original Data')

# define window size
window_size = 50

# define epochs
epochs = 1000

# apply window size to construct a batches of training data and expected prediction in labels
batches, labels = prepare_data(series, window_size)

"""
CNN Model
"""

# Model
model = CNNModel()
# Fit model with all data except last one
model.fit(batches[:-1], labels[:-1])
# Use last one to predict
model.predict(batches[-1], labels[-1])

# use as input all batches but the last one, to use as test
batches = batches[:, :, 0]
X = batches[:-1]
y = labels[:-1]

# define model
MLP_model = MLPModel(window_size)

# # train
# MLP_model.fit(X, y, epochs)
#
# # # predict
# prediction = MLP_model.predict(batches[-1])
# print("Predicted: {0}\nExpecred:  {1}".format(prediction, labels[-1]))


# simulate next steps in the series and compare with original
steps = 200
train_set = batches[:-200]
train_labels = labels[:-200]
MLP_model.fit(train_set, train_labels, epochs)
predictions = simulation_mode(batches[-200], MLP_model, steps)
print(predictions)