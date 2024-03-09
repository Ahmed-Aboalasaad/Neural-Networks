import numpy as np

class perceptron:
    def _init_(self, learning_rate, activation_function) -> None:
        "initialize learning rate, activation_function as a lambda passed through the constructor, then weights and bias with random values"
        pass

    def fit(self, x_train, y_train) -> np.ndarray:
        "returns an np array with the final weights and bias after the fitting process is over"
        pass

    def predict(self, x_test) -> np.ndarray:
        "returns an np array of predictions"
        pass

    def mean_squared_error(y_true, y_pred):
        return (1 / len(y_true)) * ((y_true - y_pred)**2).sum()