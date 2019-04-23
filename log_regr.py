import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import scipy.io


# mat = scipy.io.loadmat('monk2.mat')
# print(mat)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    def __init__(self, training_set, output_set, learning_rate, regularization_term):
        self.training_set = training_set
        self.output_set = output_set
        self.learning_rate = learning_rate
        self.regularization_term = regularization_term

        input_size = self.training_set.shape

        input_layer = np.zeros(input_size)

        # Add bias
        self.input_layer = np.c_[np.ones(input_size[0]), input_layer]

        self.input_weights = np.random.uniform(0, 0.2, input_size[1]+1)

    def train_network(self, epochs):
        for epoch in range(epochs):
            epoch_error = 0
            for row_idx in range(len(self.training_set)):
                prediction = self.forward(self.training_set[row_idx])
                error = self.cost_function(prediction, self.output_set[row_idx])
                epoch_error += error
                self.backpropagate()
                self.update_weights()

            print("Epoch: {0}, Error: {1}".format(epoch, epoch_error))

    def forward(self, input_data):
        # TODO Check if this append is correct. We did that on ARS. 1 is the bias node
        z = np.dot(np.append(input_data, 1), self.input_weights)
        prediction = sigmoid(z)

        return prediction

    def cost_function(self, prediction, real_label):
        cost = float('inf')
        if real_label == 1:
            cost = - np.log(prediction)

        elif real_label == 0:
            cost = - np.log(1 - prediction)

        return cost

    def gradient_descend(self):
        # self.input_weights = self.input_weights - self.learning_rate *
        pass

    def backpropagate(self):
        pass

    def update_weights(self):
        # Update the weights based on the learning rate
        pass


if __name__ == "__main__":
    # Retrieve Dataset
    dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                          header=None).values

    # Specify dataset parameters
    features_combination = [0, 2]
    class_index = 4
    training_data_size = 100

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
                                             output_set=output_set,
                                             learning_rate=learning_rate,
                                             regularization_term=regularization_term)

    logistic_regression.train_network(epochs=2000)
