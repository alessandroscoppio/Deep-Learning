from keras.models import Sequential
from keras.layers import Dense


class MLPModel:
    def __init__(self, input_size):
        # define model
        self.input_size = input_size
        self.model = Sequential()
        self.model.add(Dense(100, activation='relu', input_dim=self.input_size))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs):
        # fit model
        self.model.fit(X, y, epochs=epochs, verbose=1)

    def predict(self, input, expected):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size))
        prediction = self.model.predict(x_input, verbose=1)
        print("Predicted: ", prediction, "    Expected: ", expected)
