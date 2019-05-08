from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model


class MLPModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.history = None
        self.model = Sequential()
        self.model.add(Dense(100, activation='relu', input_dim=self.input_size))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0):
        # fit model
        X = X.reshape((X.shape[0], X.shape[1]))
        self.history = self.model.fit(X, y, batch_size=10, epochs=epochs, verbose=verbose)

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
        self.history = None
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', batch_input_shape=(None, self.input_size, 1)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0, batch_size=32):
        self.history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

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
        self.history = None
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.input_size, 1)))
        # self.model.add(Dropout(0.5))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0, batch_size=32):
        # fit model
        self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)


class CNNLSTMModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.history = None
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 25, 1)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0):
        # fit model
        X = X.reshape((X.shape[0], 2, 25, 1))
        self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, 2, 25, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)
