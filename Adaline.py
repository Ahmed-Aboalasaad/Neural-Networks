import numpy as np

EPSILON = 0.0001

def mean_square_error(y_true, y_pred):
        m = y_true.shape[0]
        diff_squared = (y_true - y_pred)**2
        mse = diff_squared.sum() / m
        return mse

class Adaline:
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate
        self.w_0 = 0 #bias term
        self.w_1 = 0
        self.w_2 = 0


    def fit(self, x_train, y_train):
        predictions = self.predict(x_train)
        print(type(predictions))
        print(type(y_train))
        prev_MSE = mean_square_error(y_train, predictions)
        curr_MSE = -1e20
        while abs(prev_MSE - curr_MSE) > EPSILON:
            for x1, x2, t in zip(x_train[:, 0], x_train[:, 1], y_train[:]):
                y = self.predict_single(x1, x2)
                error = t - y
                self.w_0 = self.w_0 + self.update_weight_with_GD(error, 1)
                self.w_1 = self.w_1 + self.update_weight_with_GD(error, x1)
                self.w_2 = self.w_2 + self.update_weight_with_GD(error, x2)

            predictions = self.predict(x_train)
            prev_MSE = mean_square_error(y_train, predictions)
        
    def update_weight_with_GD(self, error, x):
        return self.learning_rate * error * x


    def predict(self, X_test):
        x_w = X_test[:, 0]*self.w_1 + X_test[:, 1] * self.w_2 + self.w_0
        return x_w
    
    def predict_single(self, x1, x2):
        return x1*self.w_1 + x2*self.w_2 + self.w_0
