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
plot_series(1000, series, 'Original Data')

# define window size
window_size = 50

# define epochs
epochs = 300

# apply window size to construct a batches of training data and expected prediction in labels
batches, labels = prepare_data(series, window_size)

# # use as input all batches but the last one, to use as test
# batches = batches[:, :, 0]
# X = batches[:-1]
# y = labels[:-1]

# choose training data
train_set = batches
train_labels = labels

# initiate Models
model_mlp = MLPModel(window_size)
model_cnn = CNNModel(window_size)

# train models
model_mlp.fit(train_set, train_labels, epochs)
model_cnn.fit(train_set, train_labels, epochs)

# save models
model_mlp.save_model('mlp_{0}.h5'.format(epochs))
model_cnn.save_model('cnn_{0}.h5'.format(epochs))

# simulate next steps in the series and compare with original
starting_point_of_prediction = 1000
length_of_prediction = 200

# run simulation mode to predict the next values
predictions_cnn = simulation_mode(series, model_cnn, starting_point_of_prediction, length_of_prediction)
predictions_mlp = simulation_mode(series, model_mlp, starting_point_of_prediction, length_of_prediction)

# plot both original series and simulated predictions
y1 = series
y2 = predictions_cnn[-length_of_prediction:]
y3 = predictions_mlp[-length_of_prediction:]
x = range(starting_point_of_prediction, starting_point_of_prediction + length_of_prediction)
plt.plot(range(len(series)), y1, label="Original Series", linestyle="solid", color='blue')
plt.plot(x, y2, label="Simulated CNN", linestyle="dotted", color='red')
plt.plot(x, y3, label="Simulated MLP", linestyle="dotted", color='green')
plt.title("Predicted Values")
plt.legend()
plt.show()
