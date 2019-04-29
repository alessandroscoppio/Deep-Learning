# univariate mlp example
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import scipy.io
import matplotlib.pyplot as plt


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
    return np.asarray(batches), np.asarray(labels)


# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])
# plot_series(1000, series, 'Original Data')

batches, labels = prepare_data(series, 50)
print(batches)
print(labels)
# # define model
# model = Sequential()
# model.add(Dense(100, activation='relu', input_dim=100))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# model.fit(X, y, epochs=2000, verbose=0)
# # demonstrate prediction
# x_input = np.array([50, 60, 70])
# x_input = x_input.reshape((1, 3))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
