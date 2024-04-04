import activation_functions
import numpy as np



activation_function_mapping = {
    'Tanh' : activation_functions.Tanh,
    'Sigmoid': activation_functions.Sigmoid
}


activation_function_derivative_mapping = {
    'Tanh' : activation_functions.d_Tanh,
    'Sigmoid' : activation_functions.d_Sigmoid
}


class MultiLayerPerceptron:
    def __init__(self, 
                 input_size : int, 
                 output_size : int, 
                 number_of_hidden_layers : int, 
                 neurons_per_layer : list, 
                 learning_rate : float, 
                 activation_function, # lambda expression
                 epochs : float,
                 add_bias : bool,
                 seed : int) -> None:
        
        self.input_size = input_size
        self.output_size = output_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.activation_function = activation_function_mapping[activation_function]
        self.activation_function_derivative = activation_function_derivative_mapping[activation_function]
        self.epochs = epochs
        self.add_bias = add_bias
        np.random.seed(seed)
        ## initializing weights between all layers
        self.list_of_weights = self.__init_weights()



    def fit(self, X: np.array, y):
        ## append 1 at the end of each record if add_bias is true
        print(self.list_of_weights)
        if self.add_bias:
            X = self.__add_ones(X)
        net_outputs = []
        for i in range(self.epochs):    
            current = X
            for i in range(self.number_of_hidden_layers+1):
                
            ## forward step 1 [calculating output]
                net = current.dot(self.list_of_weights[i])
                net_outputs.append(net)
                y_ = np.vectorize(self.activation_function)(net)
                current = y_
                if i != self.number_of_hidden_layers:
                    if self.add_bias:
                        current = self.__add_ones(current)
                
                print(current)

            
            ## backward step [calculating partial errors]
            net_of_output_layer = net_outputs[len(net_outputs)-1]
            errors_in_last_layer = y - current[len(current)-1]
            output_layer_gradient = errors_in_last_layer *  self.np.vectorize(self.activation_function_derivative)(net_of_output_layer)
            for i in range(self.number_of_hidden_layers+1, -1, -1):
                pass
                


            ## forward step2 [updating weights]

        pass



    def predict(self, x):
        pass



    def __init_weights(self):
        list_of_weights = []

        neurons_per_all_layers = [self.input_size]
        neurons_per_all_layers.extend(self.neurons_per_layer)
        neurons_per_all_layers.append(self.output_size)
        shortcut = 1 if self.add_bias else 0

        for i in range(self.number_of_hidden_layers+1):
            weights_arr = np.random.random_sample((neurons_per_all_layers[i] + shortcut, neurons_per_all_layers[i+1]))
            list_of_weights.append(weights_arr)

        return list_of_weights
    

    def __add_ones(self, X):
        X_shape = X.shape
        X_with_ones =np.ones((X_shape[0],X_shape[1]+1))
        X_with_ones[:, 0:X_shape[1]] = X
        return X_with_ones

    def get_weights(self):
        return self.list_of_weights