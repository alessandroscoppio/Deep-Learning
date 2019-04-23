import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import scipy.io


# mat = scipy.io.loadmat('monk2.mat')
# print(mat)


def sigmoid(x):
    # try:
    return 1 / (1 + np.exp(-x))
    # except AttributeError:
    #     print("SHIT!")

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

    # Create training set by selecting rows and columns and copying
    training_set = dataset[:, features_combination].copy()

    # Create output class vector
    output_set = dataset[:, class_index].copy()

    return training_set, output_set


def plot(data, output):
    colors = output
    plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='viridis')
    plt.colorbar()
    plt.show()


def normalize_dataset(training_set):
    # Normalize using min-max scaling
    return pre.minmax_scale(training_set)


class LogisticRegression:

    def __init__(self, training_set, labels, learning_rate, regularization_term):
        self.training_set = training_set
        self.labels = labels
        self.learning_rate = learning_rate
        self.regularization_term = regularization_term

        input_size = self.training_set.shape

        # Add bias
        self.input_layer = np.c_[np.ones(input_size[0]), training_set]

        self.input_weights = np.random.uniform(0, 0.2, input_size[1] + 1)

    def train_network(self, epochs):
        for epoch in range(epochs):
            predictions = self.forward()
            cost = self.cost_function(predictions)
            self.backpropagate(predictions)
            self.gradient_descend()

            print("Epoch: {0}, Error: {1}".format(epoch, cost))

    def forward(self):
        # TODO Check if this append is correct. We did that on ARS. 1 is the bias node
        self.z = np.dot(self.input_layer, self.input_weights)
        prediction = sigmoid(self.z)

        return prediction

    def cost_function(self, prediction):
        cost = 0
        sum_errors = 0
        for idx, pred in enumerate(prediction):
            if self.labels[idx] == 1:
                cost = - np.log(pred)

            elif self.labels[idx] == 0:
                cost = - np.log(1 - pred)
            sum_errors += cost
        return sum_errors

    def backpropagate(self, predictions):
        # gradient = np.dot(np.dot(self.labels, (1 / predictions)) - np.dot(1 - self.labels, 1 / (1 - predictions)),
        #                   sigmoid_derivate(self.z))
        # self.gradients = np.ones_like(self.input_weights)
        self.gradients = np.dot((self.labels - predictions), self.input_layer)

    def gradient_descend(self):
        # Update the weights based on the learning rate
        # for weight in range(len(self.input_weights)):
        #     self.input_weights[weight] = self.input_weights[weight] - self.learning_rate * self.gradients[weight]
        self.input_weights = self.input_weights - self.learning_rate * self.gradients


if __name__ == "__main__":
    # Retrieve Dataset
    dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                          header=None).values

    # Specify dataset parameters
    features_combination = [0, 2]
    class_index = 4
    training_data_size = 80

    # Prepare dataset
    training_set, output_set = prepare_dataset(dataset,
                                               training_data_size=training_data_size,
                                               features_combination=features_combination,
                                               class_index=class_index)
    # Normalize dataset
    training_set = normalize_dataset(training_set)
    plot(training_set, output_set)

    # Specify training parameters
    learning_rate = 0.9
    regularization_term = 0

    # Initiate Model
    logistic_regression = LogisticRegression(training_set=training_set,
                                             labels=output_set,
                                             learning_rate=learning_rate,
                                             regularization_term=regularization_term)

    logistic_regression.train_network(epochs=2000)
