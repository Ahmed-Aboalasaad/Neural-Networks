import numpy as np
import pandas as pd
import random
import activation_functions

def mean_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred)**2) / len(y_true)


class Adaline:
    def __init__(self, learning_rate , EPSILON, include_bias) -> None:
        self.learning_rate = learning_rate
        self.EPSILON = EPSILON  ## EPSILON AKA MSE THRESHOLD
        self.weights = np.random.rand(2, 1)
        self.include_bias = include_bias
        if include_bias:
            self.bias = random.random()
        else:
            self.bias = 0

    def update_weights(self, total_error, X_train):
            self.weights[0] = self.weights[0] + ((self.learning_rate*total_error)*X_train[:, 0]).sum()
            self.weights[1] = self.weights[1] + ((self.learning_rate*total_error)*X_train[:, 1]).sum()
            if self.include_bias:
                self.bias = self.bias + self.learning_rate*total_error

            
    def fit(self, X_train, y_train) -> None:
        predictions = self.predict(X_train)
        prev_MSE = mean_squared_error(y_train, predictions)
        curr_MSE = -1e20
        while abs(prev_MSE - curr_MSE) > self.EPSILON:
            total_error = np.sum((y_train - predictions))
            self.update_weights(total_error, X_train)
            predictions = self.predict(X_train)
            prev_MSE = curr_MSE
            curr_MSE = mean_squared_error(y_train, predictions)

    def predict(self, X_test : np.ndarray) -> np.ndarray:
        y = X_test.dot(self.weights) + self.bias
        a = np.vectorize(activation_functions.identity)(y)
        return a

    def predict_single(self, x1, x2):
        return x1*self.w_1 + x2*self.w_2 + self.w_0
        
    def get_weights(self):
        return self.weights[0][0], self.weights[1][0] , self.bias