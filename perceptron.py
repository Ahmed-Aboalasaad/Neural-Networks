import numpy as np
import random

# Activation functions
EULER_CONSTANT = 2.7182818
identity = lambda y : y
ReLU = lambda y : y if y >= 0 else 0
leaky_ReLU = lambda y : y if y >= 0 else 0.1*y
sigmoid = lambda y : 1 / (1 + EULER_CONSTANT ** -y)
signum = lambda y : 1 if y >= 0 else 0
tanh = lambda y, a : (1 - EULER_CONSTANT**(-y*a)) / (1 + EULER_CONSTANT**(-y*a))

class perceptron:
    def _init_(self, learning_rate, activation_function, mse_threshold) -> None:
        "initialize learning rate, activation_function as a lambda passed through the constructor, then weights and bias with random values"
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.mse_threshold = mse_threshold
        self.weights = np.array([random.random(), random.random()])
        self.bias = 0 # should the bias be initially zero?

    def fit(self, x_train, y_train) -> np.ndarray:
        "returns an np array with the final weights and bias after the fitting process is over"
        while (True):
            output = self.predict(x_train)
            cost = y_train - output
            if (cost < self.mse_threshold):
                break

            # update weights & bias
            self.weights += cost * self.learning_rate * x_train
            self.bias += cost * self.learning_rate

        return np.concatenate([self.bias], self.weights)

    def predict(self, x) -> np.ndarray:
        "returns an np array of predictions"
        net_input = np.dot(x, self.weights) + self.bias
        return self.activation_function(net_input)