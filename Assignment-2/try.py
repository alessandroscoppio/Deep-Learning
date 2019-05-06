# multi-step encoder-decoder lstm example
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
import scipy.io


def prepare_data(data, window_size):
    batches = []
    labels = []
    for idx in range(len(data) - window_size - 2):
        batches.append(data[idx: idx + window_size])
        labels.append(np.array([data[idx + window_size], data[idx + window_size + 1]]))

    return np.array(batches), np.array(labels)

from pandas import Series
from sklearn.preprocessing import MinMaxScaler

# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])

print(series)

# prepare data for normalization
values = series
values = values.reshape((len(values), 1))
# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# normalize the dataset and print
normalized = scaler.transform(values)
print(normalized)
# inverse transform and print
inversed = scaler.inverse_transform(normalized)
print(inversed)
# define dataset
# X = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
# y = np.array([[40, 50], [50, 60], [60, 70], [70, 80]])

# reshape from [samples, timesteps] into [samples, timesteps, features]
# X = X.reshape((X.shape[0], X.shape[1], 1))
# y = y.reshape((y.shape[0], y.shape[1], 1))


# plot training set
# plot_series(1000, series, 'Original Data')

# define window size
window_size = 50

# define epochs
epochs = 300

# apply window size to construct a batches of training data and expected prediction in labels
batches, labels = prepare_data(normalized, window_size)

train_set = batches[:-1]
train_labels = labels[:-1]

train_set = train_set.reshape((train_set.shape[0], train_set.shape[1], 1))
train_labels = train_labels.reshape((train_labels.shape[0], train_labels.shape[1], 1))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(window_size, 1)))
model.add(RepeatVector(2))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(train_set, train_labels, epochs=200, verbose=2)
# demonstrate prediction
# x_input = np.array([50, 60, 70])
# x_input = x_input.reshape((1, 3, 1))

x_input = batches[-1]
x_input = x_input.reshape((1, window_size, 1))

yhat = model.predict(x_input, verbose=2)
print("prediction: ", yhat)
print("real: ", labels[-1])
print("real scaled: ", scaler.inverse_transform(labels[-1]))

x = range(1000, 1200)
y2 = predictions_lstm[-200:]
plt.plot(x, y4, linestyle="solid", color='black')
plt.title("Predicted MLP")
plt.show