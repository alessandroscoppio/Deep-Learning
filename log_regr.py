import pandas as pd
import numpy as np
from sklearn import preprocessing as pre

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

dataset = pre.minmax_scale(df.head(100)[[0, 2]])
CHECK = dataset[0]
print(df)
print(df.get_value(0, 0))



def logistic_regression(input_size=2, features_combination="02", learning_rate=0.9, regularization_term=0.7):

    input_layer = np.zeros(input_size)

    input_weights = np.random.uniform(0, 0.2, input_size)

