import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = df.iloc[:100, :]
monk2 = scipy.io.loadmat('monk2.mat')['monk2']
df.head()
df.tail()
print(df.iloc[:, 4].describe())


def normalize(input_data):
    for i in range(len(input_data[0])):
        input_data[:, i] = (input_data[:, i] - np.min(input_data[:, i])) / (
                np.max(input_data[:, i]) - np.min(input_data[:, i]))

    return input_data


def binarize(output):
    first_class = output[0]
    for i in range(len(output)):
        output.iloc[i] = 0 if output[i] == first_class else 1

    return output


def initialize_weights(n_columns):
    return np.random.randn(n_columns, 1) / np.sqrt(n_columns)  # layer initialization


def initialize(data, columns=[0, 2], output_col=4, n_rows=100):
    """ Load the input of the NN. Columns must be an array """
    # Create array of indexes to take random sample
    indices = [i for i in range(len(data))]
    np.random.shuffle(indices)
    train_indices = indices[:n_rows]
    test_indices = indices[n_rows:]

    # When working with a DB that is not numpy array transform it
    if type(data) != np.ndarray:
        # If the column with the class is not an integer value, binarize it
        if type(data.iloc[0, output_col]) != int:
            output = np.array(binarize(data.iloc[:, output_col]))
        else:
            output = np.array(data.iloc[:, output_col])

        data = np.array(data)
    else:
        output = data[:, output_col]

    input_data = normalize(data[:, columns])

    # Add bias as input
    temp = np.ones((len(input_data), len(input_data[0]) + 1))
    temp[:, :len(input_data[0])] = input_data
    input_data = temp
    # Add bias node
    w = initialize_weights(len(input_data[0]))

    o_train = output[train_indices]
    i_train = input_data[train_indices]

    o_test = output[test_indices]
    i_test = input_data[test_indices]

    assert len(input_data[0]) == len(w)

    return w, o_train, i_train, o_test, i_test


def sigmoid(z):
    return 1 / (1 + math.exp(- z))


def partial_der_sigmoid(w, x):
    """ dh(z)/dz = sigmoid * (1 - sigmoid) """
    sig = h(w, x)
    return sig * (1 - sig)


def h(w, x):
    """
    Calculate the activation function
    :param w: Weights
    :param x: Input of the network
    :return: Value of the activation
    """
    return sigmoid(np.matmul(w.T, x))
    # y = max(y, 0.0001)
    # y = min(y, 0.9999)
    # return y


"""
    W = W - learning_rate * gradient_W

    gradient_W = derivative of likelihood respect to W

    dL/dW = dL/dh * dh(z)/dW

    dL/dh(z) = - (y/h(z) - (1 - y)/(1 - h(z))

    dh(z)/dW = dh(z)/dz * dz/dW

    dz/dW = input
"""


def partial_der_predictor_respect_to_weight_sigmoid(w, x, j):
    """
    dh(z)/dWj = dh(z)/dz * dz/dWj

    dh(z)/dz = sigmoid * (1 - sigmoid)
    dz/dWj = x_j

    dh(z)/dWj = x_j * sigmoid * (1 - sigmoid)
    :param w: Weights
    :param x: Inputs
    :param j: positions of the weight for the partial derivative
    :return:
    """
    return x[j] * partial_der_sigmoid(w, x)


def partial_derivative_likelihood(y, x, w):
    """ dL / dh """
    prediction = h(w, x)
    return - (y / prediction - (1 - y) / (1 - prediction))


def update_weights(learning_rate, y, x, w, regularization, weight_decay):
    for pos, value in enumerate(w[:-1]):
        partial_der_term = partial_derivative_likelihood(y, x, w) * \
                           partial_der_predictor_respect_to_weight_sigmoid(w, x, pos)
        if regularization:
            partial_der_term += weight_decay * w[pos]

        w[pos] -= learning_rate * partial_der_term

    # Update bias node
    pos = -1
    w[pos] -= learning_rate * partial_derivative_likelihood(y, x, w) * partial_der_sigmoid(w, x)


def cost(y, x, w):
    return abs(y - h(w, x))


def train(learning_rate, w, input_train, output_train, regularization, allowed_error=0.05, weight_decay=None):
    """
    Train the perceptron, maximum of 1000 batches or until allowed error is achieved
    :param learning_rate:
    :param w: Weights
    :param input_train: Input of the network
    :param output_train: Output that the network should predict
    :param regularization: Boolean value
    :param allowed_error: Error that makes the training stop
    :param weight_decay: parameter for regularization term
    :return: Number of batches needed to achieve the allowed error
    """
    error = 1
    batch = 0
    while error > allowed_error and batch < 1000:
        batch += 1
        error = 0
        for i in range(len(input_train)):
            x = input_train[i, :]
            update_weights(learning_rate, output_train[i], x, w, regularization, weight_decay)
            error += cost(output_train[i], x, w)
        error /= len(input_train)
        # print("Error", error, "batch", batch)

    return batch


def test(w, input_test, output_test, load_plot=True):
    """
    Test the performance of the perceptron
    :param w: Weights
    :param input_test: Input data, numpy ndarray
    :param output_test: Output data, numpy ndarray
    :param load_plot: Flag to load the scatter plot
    :return: sum of the error, number of errors
    """
    error = 0
    n_errors = 0
    pred_0, pred_1, label_0, label_1 = [], [], [], []

    for i in range(len(input_test) - 1):
        #         print(i)
        x = input_test[i, :]
        #         print(x)
        error += cost(output_test[i], x, w)
        prediction = 0 if h(w, x) < 0.5 else 1
        if prediction != output_test[i]:
            n_errors += 1

        if load_plot:
            if prediction == 0:
                pred_0.append(x)
            else:
                pred_1.append(x)

            if output_test[i] == 0:
                label_0.append(x)
            else:
                label_1.append(x)

    if load_plot:
        plt.scatter([x[0] for x in pred_0], [x[1] for x in pred_0], c='r', label='Predicted class 0')
        plt.scatter([x[0] for x in label_0], [x[1] for x in label_0], c='r', label='Label class 0', marker='x', s=100)
        plt.scatter([x[0] for x in pred_1], [x[1] for x in pred_1], c='b', label='Predicted class 1')
        plt.scatter([x[0] for x in label_1], [x[1] for x in label_1], c='b', label='Label class 1', marker='x', s=100)
        x = np.linspace(0, 1, 100)
        y = - (w[0] * x + w[2]) / w[1]
        plt.plot(x, y, label='Boundary')

        plt.legend()

    return error / len(input_test), n_errors


def study_learning_rate(data, input_col, output_col, n_elems, allowed_error=0.05, regularization=False,
                        weight_decay=0.001, n_learning_rates=10):
    learning_rates = np.linspace(0.001, 1, n_learning_rates)
    errors, n_errors_h, batches, accuracies = [], [], [], []

    for learning_rate in learning_rates:
        weights, output_train, input_train, output_test, input_test = initialize(data, input_col, output_col, n_elems)
        n_batches = train(learning_rate, weights, input_train, output_train, allowed_error=allowed_error,
                          regularization=regularization, weight_decay=weight_decay)
        error, n_errors = test(weights, input_test, output_test, load_plot=False)
        errors.append(error)
        n_errors_h.append(n_errors)
        batches.append(n_batches)
        accuracies.append(100 - round(100 * n_errors / len(input_test), 3))
        print("Learning rate:", learning_rate)
        print("Number of batches needed to converge", n_batches)
        # print("Obtained error in testing", error)
        print(n_errors, "errors when testing", len(input_test), "data points. Accuracy =",
              100 - round(100 * n_errors / len(input_test), 3), "%")
        print()

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("Learning rate")
    ax1.set_ylabel("Average error in test set", color=color)
    ax1.plot(learning_rates, errors, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("Number of errors", color=color)
    ax2.plot(learning_rates, n_errors_h, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Effect of learning rate" +
              (" with regularization, weight decay" + str(weight_decay) if regularization else ""))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.figure()
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("Learning rate")
    ax1.set_ylabel("Number of batches needed", color=color)
    ax1.plot(learning_rates, batches, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("Accuracy %", color=color)
    ax2.plot(learning_rates, accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Effect of learning rate" +
              (" with regularization, weight decay" + str(weight_decay) if regularization else ""))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()

    return learning_rates[np.argmin(errors)]


def study_random_columns():
    columns = [0, 1, 2, 3]
    tested = []
    learning_rate = 0.001
    for i in range(4):
        while columns[:2] in tested:
            np.random.shuffle(columns)
        weights, output_train, input_train, output_test, input_test = initialize(df, columns[:2])
        train(learning_rate, weights, input_train, output_train, regularization=False)
        test(weights, input_test, output_test, load_plot=True)
        plt.title("Columns " + ' ,'.join([str(i) for i in columns[:2]]))
        plt.xlabel("Column " + str(columns[0]))
        plt.ylabel("Column " + str(columns[1]))
        plt.show()
        tested.append(columns[:2])


def test_learning_rate(data, n_learning_rates=10, output_column=None, regularization=True, weight_decay=0.01):
    """
    Test the learning rate for a specific database. When the DB is not a numpy ndarray the output column is mandatory.
    :param data: DB to study
    :param n_learning_rates: Number of learning rates to study. Spaced linearly from 0.001 to 1.
    :param output_column: When the DB is a pandas object the output columns is mandatory.
    :param regularization: Flag that activates the regularization term
    :param weight_decay: value of the weight decay for the regularization term
    :return:
    """

    if type(data) != np.ndarray and output_column is None:
        raise Exception("When the DB is not a numpy ndarray the output column is mandatory.")

    if type(data) != np.ndarray:
        input_columns = [i for i in range(len(data.iloc[0]) - 1)]
    else:
        input_columns = [i for i in range(len(data[0]) - 1)]
    np.random.shuffle(input_columns)
    input_columns = input_columns[:2]
    if output_column is None:
        output_column = len(data[0]) - 1
    n_rows = int(0.8 * len(data))

    best_learning_rate = study_learning_rate(data, input_columns, output_column, n_rows,
                                             n_learning_rates=n_learning_rates)

    weights, output_train, input_train, output_test, input_test = initialize(data, input_columns, output_column,
                                                                             n_rows)

    n_batches = train(best_learning_rate, weights, input_train, output_train,
                      regularization=regularization, weight_decay=weight_decay)
    error, n_errors = test(weights, input_test, output_test, load_plot=True)

    print("Using the best learning rate found:", best_learning_rate)
    print("Number of batches needed to converge (if 1000, it didn't converged)", n_batches)
    print("Sum of all the errors, being error = abs(y - predicted)", error)
    print(n_errors, "errors when testing", len(input_test), "data points. Accuracy =",
          100 - round(100 * n_errors / len(input_test), 3), "%")

    plt.title("Result using best learning rate found: " + str(best_learning_rate))
    plt.xlabel("Column " + str(input_columns[0]))
    plt.ylabel("Column " + str(input_columns[1]))
    plt.show()


test_learning_rate(df, output_column=4)
test_learning_rate(monk2, n_learning_rates=5)

# study_learning_rate()
# study_random_columns()
