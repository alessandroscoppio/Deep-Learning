import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except OverflowError:
        return 1e-13


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
        plt.plot(range(len(cost_history[i])), cost_history[i], label="Learning rate {0}".format(learning_rates[i]))
    plt.legend()
    plt.show()


def get_accuracy(test_labels, test_pred):
    success = 0
    for i in range(len(test_labels)):
        if test_labels[i] == test_pred[i].round():
            success += 1
    return success / len(test_labels)


def learning_rate_experiments():
    learning_rate = [0.1, 0.2, 0.3, 0.4]
    cost_history_per_lr = []
    for lr in learning_rate:
        logistic_regression = LogisticRegression(training_set=training_set,
                                                 labels=training_labels,
                                                 learning_rate=lr)

        logistic_regression.train_network(epochs=1000)
        cost_history_per_lr.append(logistic_regression.cost_history)
        test_pred = logistic_regression.test_network(test_set)
        print(get_accuracy(test_labels, test_pred) * 100)

    plot_cost_history(learning_rate, cost_history_per_lr)


class LogisticRegression:
    def __init__(self, training_set, labels, learning_rate):
        self.training_set = training_set
        self.labels = labels
        self.learning_rate = learning_rate

        self.cost_history = []

        input_size = self.training_set.shape

        self.input_layer = training_set
        self.input_weights = np.random.uniform(0, 0.4, input_size[1])

    def forward(self, input):
        z = np.dot(input, self.input_weights)
        prediction = np.array(sigmoid(z), dtype=float)

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
        for idx in range(self.input_weights.shape[0]):
            self.gradients[idx] = np.dot((predictions - self.labels), self.input_layer[:, idx])

    def gradient_descend(self):
        self.input_weights = self.input_weights - (self.learning_rate * self.gradients) / self.input_layer.shape[0]

    def train_network(self, epochs):
        for epoch in range(epochs):
            predictions = self.forward(self.input_layer)
            cost = self.cost_function(predictions)
            self.cost_history.append(cost)
            self.backpropagation(predictions)
            self.gradient_descend()

            if np.abs(cost) < 1e-11:
                print("Convergence reached by threshold")
                return

            print("Epoch: {0}, Cost: {1}".format(epoch, cost))

            # Uncomment these two lines to see progression of decision boundary while
            # network is being trained
            # if not epoch % 100:
            #     plot_decision_boundary(training_set, training_labels, self.test_network)

    def test_network(self, test_set):
        return self.forward(test_set)


if __name__ == "__main__":
    # For repeatability
    np.random.seed(12)

    # Retrieve Dataset
    dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                          header=None).values

    monk2 = scipy.io.loadmat('Assignment-1/monk2.mat')['monk2']

    for row in dataset:
        if row[4] == 'Iris-setosa':
            row[4] = 0.0
        elif row[4] == 'Iris-versicolor':
            row[4] = 1.0

    features_combination = [0, 1]

    dataset = dataset[:100]
    np.random.shuffle(dataset)

    # Convert labels to same type as dataset
    labels = dataset[:, -1].astype(np.float64)

    norm_dataset = dataset[:, features_combination].astype(np.float64)

    # Training set is 80%
    training_set = norm_dataset[:80]
    training_labels = labels[:80]

    # Test set is 20%
    test_set = norm_dataset[-20:]
    test_labels = labels[-20:]

    # Specify training parameters
    learning_rate = 0.1

    # Monk database
    # Uncomment this lines to test monk2 dataset
    # training_set = monk2[:345, features_combination]
    # training_labels = monk2[:345, -1]
    # test_set = monk2[345:, features_combination]
    # test_labels = monk2[345:, -1]

    # Uncomment to perform learning experiments if desired
    learning_rate_experiments()

    # Initiate Model
    logistic_regression = LogisticRegression(training_set=training_set,
                                             labels=training_labels,
                                             learning_rate=learning_rate)

    # Train model for 10000 epochs
    logistic_regression.train_network(epochs=10000)

    # Plot cost history
    plot_cost_history([learning_rate], [logistic_regression.cost_history])

    # Retrieve prediction for the test_set by the trained network
    test_pred = logistic_regression.test_network(test_set)

    # Get Accuracy
    accuracy = get_accuracy(test_labels, test_pred) * 100
    print("Prediction Accuracy {0}%".format(accuracy))

    # Plot decision boundary
    plot_decision_boundary(training_set, training_labels, logistic_regression.test_network)
