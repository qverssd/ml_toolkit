import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_interations=1000):
        self.learning_rate = learning_rate
        self.n_interations = n_interations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_spamples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_interations):
            y_predicted = np.dot(x, self.weights) + self.bias
            dw = (1 / n_spamples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_spamples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias