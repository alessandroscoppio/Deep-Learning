import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from models import *


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


def run_model(model, model_name):
    # apply window size to construct a batches of training data and expected prediction in labels
    batches, labels = prepare_data(series, window_size)

    # Load model from memory (if already trained once)
    model.load_model('saved-models/{0}_1500.h5'.format(model_name))

    # Train model
    # model.fit(batches, labels, epochs, 2)

    # Save model
    # model.save_model('cnn_{0}.h5'.format(epochs))

    # Run simulation model and retrieve predictions
    predictions = simulation_mode(series, model, window_size, starting_point_of_prediction, length_of_prediction)

    mse = mean_squared_error(series_test, predictions[-200:])
    mae = mean_absolute_error(series_test, predictions[-200:])
    rmse = np.sqrt(mse)
    r2 = r2_score(series_test, predictions[-200:])

    print("MSE of {0} model: {1}".format(model_name, mse))
    print("MAE of {0} model: {1}".format(model_name, mae))
    print("RMAE of {0} model: {1}".format(model_name, rmse))
    print("R2score of {0} model: {1}".format(model_name, r2))
    plt.plot(predictions, linewidth=0.5, linestyle="solid", color='blue')
    plt.plot(range(1000, 1200), series_test, linewidth=0.5, linestyle="solid", color='red')
    plt.title(model_name)
    plt.show()



# define dataset
series = np.array(scipy.io.loadmat('Xtrain.mat')['Xtrain'])
series_test = np.array(scipy.io.loadmat('Xtest.mat')['Xtest'])

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(series)
# series = scaler.transform(series)
#
# series_test = scaler.transform(series_test)
# define window size
window_size = 50

# define epochs
epochs = 100

# initiate Models
# model_mlp = MLPModel(window_size)
model_cnn = CNNModel(window_size)
# model_lstm = LSTMModel(window_size)
model_cnnlstm = CNNLSTMModel(window_size)

# simulate next steps in the series and compare with original
starting_point_of_prediction = 1000
length_of_prediction = 200

# Run any of the models
# run_model(model_mlp, "mlp")  # Change to any of the other models from above
run_model(model_cnn, "cnn")  # Change to any of the other models from above
# run_model(model_lstm, "lstm")  # Change to any of the other models from above
run_model(model_cnnlstm, "cnnlstm")  # Change to any of the other models from above

# Run experiments. Uncomment to execute
# experiment_with_batch_and_window_size()
# experiment_normalization()

