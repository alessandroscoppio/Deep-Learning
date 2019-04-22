import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
import scipy.io

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

dataset = pre.minmax_scale(df.head(100)[[0, 2]])
CHECK = dataset[0]
# print(df)
# print(df.get_value(0, 0))

mat = scipy.io.loadmat('monk2.mat')
print(mat)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:

    def __init__(self, dataset, features_combination=[0, 2], learning_rate=0.9, regularization_term=0.7):

        self.nr_features = len(features_combination)

        for i in features_combination:
            self.training_set = dataset[]

        input_size = input.shape

        input_layer = np.zeros(input_size)

        input_weights = np.random.uniform(0, 0.2, input_size)

        z = np.multiply(input_layer, input_weights)

        prediction = sigmoid(z)

