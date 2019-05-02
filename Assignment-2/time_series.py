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
    return np.array(batches), np.array(labels)


# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])
# plot training set
plot_series(1000, series, 'Original Data')

# define window size
window_size = 50

# apply window size to construct a batches of training data and expected prediction in labels
batches, labels = prepare_data(series, window_size)

# use as input all batches but the last one, to use as test
batches = batches[:, :, 0]
X = batches[:-1]
y = labels[:-1]


# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=(window_size)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=1)
# demonstrate prediction
x_input = batches[-1]
x_input = x_input.reshape((1, window_size))
prediction = model.predict(x_input, verbose=1)
print("Predicted: ", prediction, "    Expected: ", labels[-1])
