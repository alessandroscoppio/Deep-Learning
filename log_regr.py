import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
import scipy.io


# mat = scipy.io.loadmat('monk2.mat')
# print(mat)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivate(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


def replace_class_on_dataset(dataset):
    class_mapping = {}
    unique_count = 0
    for row in dataset:
        if not row[4] in class_mapping:
            class_mapping[row[4]] = unique_count
            row[4] = unique_count
            unique_count += 1
        else:
            row[4] = class_mapping[row[4]]

    return dataset


class LogisticRegression:

    def __init__(self, dataset, training_data_size, features_combination, learning_rate=0.9, regularization_term=0.7):
        # Number of dataset features
        self.nr_features = len(features_combination)

        # Truncate dataset
        self.dataset = dataset[:training_data_size]

        # Replace classes on the last column with number
        self.dataset = replace_class_on_dataset(self.dataset)

        # Create training set by selecting rows and columns and copying
        self.training_set = self.dataset[:, features_combination].copy()

        # Normalize using min-max scaling
        self.training_set = pre.minmax_scale(self.training_set)

        self.learning_rate = learning_rate
        self.regularization_term = regularization_term

        input_size = self.training_set.shape

        input_layer = np.zeros(input_size)

        # Add bias
        self.input_layer = np.c_[np.ones(input_size[0]), input_layer]

        self.input_weights = np.random.uniform(0, 0.2, input_size[1]+1)

    def forward(self):

        z = np.dot(self.input_layer, self.input_weights)
        prediction = sigmoid(z)

        return prediction

    def gradient_descend(self):
        # self.input_weights = self.input_weights - self.learning_rate *
        pass

    def cost_function(self, prediction, real_label):

        if real_label == 1:
            cost = - np.log(prediction)

        elif real_label == 0:
            cost = - np.log(1 - prediction)

        return cost


if __name__ == "__main__":
    # Retrieve Dataset
    dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                          header=None).values
    # Specify parameters
    features_combination = [0, 2, 4]  # 4 is class
    learning_rate = 0.9
    regularization_term = 0

    # Initiate Model
    logistic_regression = LogisticRegression(dataset,
                                             training_data_size=100,
                                             features_combination=features_combination,
                                             learning_rate=learning_rate,
                                             regularization_term=regularization_term)
    logistic_regression.forward()
