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
                 hidden_layers_number : int, 
                 neurons_per_layer : list, 
                 learning_rate : float, 
                 activation_function, # lambda expression
                 epochs : float,
                 add_bias : bool,
                 seed : int) -> None:
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_number = hidden_layers_number
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.activation_function = activation_function_mapping[activation_function]
        self.activation_function_derivative = activation_function_derivative_mapping[activation_function]
        self.epochs = epochs
        self.add_bias = add_bias
        np.random.seed(seed)
        
        ## initializing weights between all layers
        self.list_of_weights = self.__init_weights()



    def fit(self, X: np.array, y: np.array):
        self.data_size = X.shape[0]
        self.list_of_weights = [np.array([[0.21, -0.4], [0.15, 0.1], [-0.3, 0.24]]), np.array([[-0.2], [0.3], [-0.4]])]
        ## append 1 at the end of each record if add_bias is true
        if self.add_bias:
            X = self.__add_ones(X)
        
        # List of lists: each of which has the net outputs of a layer
        net_outputs = []
        for j in range(self.epochs):    
            current = X

            ### Feeding Forward [calculating output]
            for i in range(self.hidden_layers_number + 1):
                current_net_output = current.dot(self.list_of_weights[i]) # input * Weights 
                net_outputs.append(current_net_output)

                activation = self.activation_function(current_net_output)
                current = activation
                # if it's not the output layer
                if i != self.hidden_layers_number and self.add_bias:
                    current = self.__add_ones(current)

            
            ### backward step [calculating cost gradients with respect to neurons]
            output_layer_error = y - current  # error calculated in output layer

            # initializing gradients array
            gradients = [np.zeros((1, n_neurons)) for n_neurons in self.neurons_per_layer]
            gradients.append(np.zeros((1, self.output_size)))

            # Output-layer gradient
            gradients[self.hidden_layers_number][:] = (output_layer_error * self.activation_function_derivative(net_outputs[self.hidden_layers_number])).sum(axis=0)/self.data_size
            # Calculate cost gradients with respect to neurons in hidden layers
            for i in range(self.hidden_layers_number - 1, -1, -1):
                gn_t_w = gradients[i+1].dot(self.list_of_weights[i+1][:-1].T)
                gradients[i][:] = gn_t_w * self.activation_function_derivative(net_outputs[i]).sum(axis=0)
                
                
            ## forward step2 [updating weights]
            



    def predict(self, x):
        pass



    def __init_weights(self):
        '''
        Initializes the list of weights with numpy arrays filled with a proper number of random weights
        '''
        list_of_weights = []

        neurons_per_all_layers = [self.input_size]
        neurons_per_all_layers.extend(self.neurons_per_layer)
        neurons_per_all_layers.append(self.output_size)
        shortcut = 1 if self.add_bias else 0

        for i in range(self.hidden_layers_number+1):
            weights_arr = np.random.rand(neurons_per_all_layers[i] + shortcut, neurons_per_all_layers[i+1])
            list_of_weights.append(weights_arr)

        return list_of_weights
    

    def __add_ones(self, X):
        '''
        Adding ones weights to compute the bias (represented as extra input)
        '''
        X_shape = X.shape
        X_with_ones =np.ones((X_shape[0],X_shape[1]+1))
        X_with_ones[:, 0:X_shape[1]] = X
        return X_with_ones

    def get_weights(self):
        return self.list_of_weights
