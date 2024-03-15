import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, bias_flag=True):
        self.learning_rate = learning_rate 
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.bias_flag=bias_flag
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize weights and bias
        # we want weights to have the length of the number of features because we have w for each feature
        self.weights = np.random.random(n_features)
        
        self.bias = 0

        max_label = np.max(y) # Finds the maximum label in the 'y' array (original labels)
        y_ = np.array([1 if i == max_label else -1 for i in y]) # Transforms labels in 'y' to binary
        
        for iter in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                
                # Perceptron update rule
                update = self.learning_rate * (y_predicted - y_[idx])
                
                self.weights -= update * x_i
                if self.bias_flag:
                    self.bias -= update
        
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted
    
    def activation_function(self, x):
        return np.where(x>=0, 1, -1)

