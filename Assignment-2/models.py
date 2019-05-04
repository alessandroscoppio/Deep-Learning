from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model


class MLPModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.model = Sequential()
        # self.model.add(Dense(100, activation='relu', input_shape=(self.input_size, 1)))
        self.model.add(Dense(100, activation='relu', input_dim=self.input_size))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs):
        # fit model
        X = X.reshape((X.shape[0], X.shape[1]))
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)


class CNNModel:
    def __init__(self, input_size):
        self.input_size = input_size
        self.model = Sequential()
        # self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.input_size, 1)))
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', batch_input_shape=(None, self.input_size, 1)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs):
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, x):
        x_input = x.reshape((1, self.input_size, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction[0]

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)


class LSTMModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.model = Sequential()
        self.model.add(Dense(100, activation='relu', input_dim=self.input_size))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs):
        # fit model
        X = X.reshape((X.shape[0], X.shape[1]))
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)

