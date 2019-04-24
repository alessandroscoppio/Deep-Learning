import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
from sklearn import base
import matplotlib.pyplot as plt
import scipy.io


# mat = scipy.io.loadmat('monk2.mat')
# print(mat)

def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except OverflowError:
        return 1e-13


def sigmoid_derivate(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


def replace_class_on_dataset(dataset, class_index):
    class_mapping = {}
    unique_count = 0
    for row in dataset:
        if not row[class_index] in class_mapping:
            class_mapping[row[class_index]] = unique_count
            row[class_index] = unique_count
            unique_count += 1
        else:
            row[class_index] = class_mapping[row[class_index]]

    return dataset


def prepare_dataset(dataset, training_data_size, features_combination, class_index):
    # Truncate dataset
    dataset = dataset[:training_data_size]

    # Replace classes on the last column with number
    dataset = replace_class_on_dataset(dataset, class_index)
    dataset = dataset.astype(np.float64)

    # Create training set by selecting rows and columns and copying
    training_set = dataset[:, features_combination].copy()

    # Create output class vector
    output_set = dataset[:, class_index].copy()

    return training_set, output_set


def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def plot_cost_history(learning_rates, cost_history):
    for i in range(len(learning_rates)):
        plt.plot(range(len(cost_history[i])), cost_history[i], label="lr {0}".format(learning_rates[i]))
    plt.legend()
    plt.show()


def get_accuracy(test_labels, test_pred):
    success = 0
    for i in range(len(test_labels)):
        if test_labels[i] == test_pred[i].round():
            success += 1
    return success / len(test_labels)


def normalize_dataset(training_set):
    # Normalize using min-max scaling
    return pre.minmax_scale(training_set)


class LogisticRegression:

    def __init__(self, training_set, labels, learning_rate, regularization_term):
        self.training_set = training_set
        self.labels = labels
        self.learning_rate = learning_rate
        self.regularization_term = regularization_term

        self.cost_history = []

        input_size = self.training_set.shape

        # Add bias
        # self.input_layer = np.c_[np.ones(input_size[0]), training_set]
        # No bias
        self.input_layer = training_set

        # self.input_weights = np.random.uniform(0, 0.2, input_size[1] + 1)
        # No bias
        self.input_weights = np.random.uniform(0, 0.4, input_size[1])

    def forward(self, input):
        # np.random.shuffle(input)
        self.z = np.zeros(input.shape)
        self.z = np.dot(input, self.input_weights)
        prediction = np.array(sigmoid(self.z), dtype=float)

        return prediction

    def cost_function(self, prediction):
        cost = 0
        sum_errors = 0
        for idx, pred in enumerate(prediction):
            if self.labels[idx] == 1:
                cost = - self.labels[idx] * np.log(pred + 1e-13)

            elif self.labels[idx] == 0:
                cost = - (1 - self.labels[idx]) * np.log(1 - pred + 1e-13)
            sum_errors += cost
        return sum_errors / len(prediction)
        # return sum_errors

    def backpropagation(self, predictions):

        self.gradients = np.zeros(self.input_weights.shape[0])

        # gradient_cost_predict = np.divide(self.labels, predictions) - np.divide((1 - self.labels), (1 - predictions))
        # gradient_predict_weight = np.dot(sigmoid_derivate(self.z), self.input_layer)
        # gradient_cost_weight = np.dot(gradient_cost_predict, gradient_predict_weight)
        #
        # self.gradients = gradient_cost_weight

        for idx in range(self.input_weights.shape[0]):
            # self.gradients[idx] = np.dot((self.labels - predictions), self.input_layer[:, idx])
            self.gradients[idx] = np.dot((predictions - self.labels), self.input_layer[:, idx])

        # self.gradients = (self.labels - predictions) * self.input_layer

        # self.gradients = np.dot((self.labels - predictions), self.input_layer) / len(self.labels)
        # self.gradients = np.dot((self.labels - predictions), self.input_layer)

    def gradient_descend(self):
        # Update the weights based on the learning rate
        # for weight in range(len(self.input_weights)):
        #     self.input_weights[weight] = self.input_weights[weight] - self.learning_rate * self.gradients[weight]
        self.input_weights = self.input_weights - (self.learning_rate * self.gradients)

        # Try! Normalize weights each iter
        # self.input_weights = self.input_weights / max(self.input_weights)

    def train_network(self, epochs):
        for epoch in range(epochs):
            predictions = self.forward(self.input_layer)
            cost = self.cost_function(predictions)
            self.cost_history.append(cost)
            self.backpropagation(predictions)
            self.gradient_descend()

            if np.abs(cost) < 0.02:
                print("Convergence reached by threshold")
                return

            print("Epoch: {0}, Cost: {1}".format(epoch, cost))
            # if not epoch % 1000:
            #     plot_decision_boundary(training_set, training_labels, logistic_regression.test_network)

    def test_network(self, test_set):
        return self.forward(test_set)


def learning_rate_experiments():
    learning_rate = [0.02, 0.04, 0.06, 0.08, 0.1]
    cost_history_per_lr = []
    for lr in learning_rate:
        logistic_regression = LogisticRegression(training_set=training_set,
                                                 labels=training_labels,
                                                 learning_rate=lr,
                                                 regularization_term=regularization_term)

        logistic_regression.train_network(epochs=300)
        cost_history_per_lr.append(logistic_regression.cost_history)
    plot_cost_history(learning_rate, cost_history_per_lr)


if __name__ == "__main__":
    # For repeatability
    np.random.seed(12)
    # Retrieve Dataset
    dataset = pd.read_csv('../iris.data',
                          header=None).values

    monk2 = scipy.io.loadmat('monk2.mat')['monk2']

    for row in dataset:
        if row[4] == 'Iris-setosa':
            row[4] = 0.0
        elif row[4] == 'Iris-versicolor':
            row[4] = 1.0

    # # Specify dataset parameters
    # features_combination = [0, 2]
    # class_index = 4
    # training_data_size = 100
    #
    # # Prepare dataset
    # training_set, output_set = prepare_dataset(dataset,
    #                                            training_data_size=training_data_size,
    #                                            features_combination=features_combination,
    #                                            class_index=class_index)

    features_combination = [0, 2]

    # Normalize dataset
    dataset = dataset[:100]
    np.random.shuffle(dataset)

    labels = dataset[:, -1].astype(np.float64)

    norm_dataset = normalize_dataset(dataset[:, features_combination])

    training_set = norm_dataset[:80]
    training_labels = labels[:80]
    test_set = norm_dataset[-20:]
    test_labels = labels[-20:]
    # Specify training parameters
    learning_rate = 0.05
    regularization_term = 0

    # Monk database
    # training_set = normalize_dataset(monk2[:345, [0, 1]])
    # training_labels = monk2[:345, -1]
    # test_set = monk2[345:, [0, 1]]
    # test_labels = monk2[345:, -1]

    # learning_rate_experiments()
    # Initiate Model
    logistic_regression = LogisticRegression(training_set=training_set,
                                             labels=training_labels,
                                             learning_rate=learning_rate,
                                             regularization_term=regularization_term)

    logistic_regression.train_network(epochs=10000)
    test_pred = logistic_regression.test_network(test_set)
    print("Prediction Accuracy {0}%".format(get_accuracy(test_labels, test_pred) * 100))


    plot_decision_boundary(training_set, training_labels, logistic_regression.test_network)
