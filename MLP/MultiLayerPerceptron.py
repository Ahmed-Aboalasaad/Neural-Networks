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
                 neurons_per_layer : list,  # number of neurons in each hidden layer
                 learning_rate : float, 
                 activation_function, # a string that get parsed to a lambda expression
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
        self.weights = self.__init_weights()  # initializing weights between ALL layers



    def fit(self, X: np.array, y: np.array):
        self.data_size = X.shape[0]
        # self.weights = [np.array([[0.21, -0.4], [0.15, 0.1], [-0.3, 0.24]]), np.array([[-0.2], [0.3], [-0.4]])]
        ## append 1 at the end of each record if add_bias is true (as if it's an additional neuron)
        if self.add_bias:
            X = self.__add_ones_nD(X)
        
        for j in range(self.epochs):
            for record, label in zip(X, y):    
                previous_activation = record
                net_inputs = []
                activations = [record]
                ### 1- Feeding Forward [calculating net inputs & activations]
                for i in range(self.hidden_layers_number + 1):
                    net_input = previous_activation.dot(self.weights[i])
                    net_inputs.append(net_input)
                    activation = self.activation_function(net_input)
                    previous_activation = activation

                    # Append the activations if they're not the output layer activations
                    if i != self.hidden_layers_number:
                        # Add ones as if the bias is an additional neuron (if the user asked to)
                        if self.add_bias:
                            previous_activation = self.__add_ones_1D(previous_activation)
                        activations.append(previous_activation)
                
                ### 2- Backward step
                # initialize gradients array
                gradients = [np.zeros((1, n_neurons)) for n_neurons in self.neurons_per_layer]
                gradients.append(np.zeros((1, self.output_size)))
                output_layer_error = label - previous_activation
                
                # Cost Gradient with respect to output-layer neurons
                gradients[self.hidden_layers_number][:] = ((output_layer_error) * self.activation_function_derivative(net_inputs[self.hidden_layers_number])).mean(axis=0)
                
                # Cost Gradient with respect to hidden-layers neurons
                for i in range(self.hidden_layers_number - 1, -1, -1):
                    if self.add_bias:
                        gn_t_w = gradients[i+1].dot(self.weights[i+1][:-1].T)
                    else :
                        gn_t_w = gradients[i+1].dot(self.weights[i+1].T)
                    gradients[i][:] = gn_t_w * self.activation_function_derivative(net_inputs[i])
                print('gradients')
                print(gradients)

                ## 3- Updating Weights
                neurons_count = []
                neurons_count.extend(self.neurons_per_layer)
                neurons_count.append(self.output_size)
                for i, activation in enumerate(activations):
                    for j in range(neurons_count[i]):
                        self.weights[i][:, j] = self.weights[i][:, j] + self.learning_rate * gradients[i][:, j] * activation
                print('weights')
                print(self.weights)

    def predict(self, X):
        if self.add_bias:
            X = self.__add_ones_nD(X)
        local_current = X

        for i in range(self.hidden_layers_number+1):
            net = local_current.dot(self.weights[i])
            activation = self.activation_function(net)
            local_current = activation
            if i != self.hidden_layers_number and self.add_bias:
                local_current = self.__add_ones_nD(local_current)
        return local_current

    def __init_weights(self):
        '''
        Initializes the weights list with np arrays filled with a proper number of random weights each
        '''
        weights = []

        # [2, 4, 1] means 2 neurons in input layer, 4 neurons in the single hidden layer, and 1 output neuron in the output layer
        neurons_count = [self.input_size]
        neurons_count.extend(self.neurons_per_layer)
        neurons_count.append(self.output_size)
        shortcut = 1 if self.add_bias else 0

        for i in range(self.hidden_layers_number+1):
            weights_arr = np.random.rand(neurons_count[i] + shortcut, neurons_count[i+1])
            weights.append(weights_arr)

        return weights
    
    def __add_ones_nD(self, X):
        '''
        Adding ones weights to compute the bias (represented as extra input)
        '''
        X_shape = X.shape
        X_with_ones =np.ones((X_shape[0],X_shape[1]+1))
        X_with_ones[:, 0:X_shape[1]] = X
        return X_with_ones
    
    def __add_ones_1D(self, X):
        '''
        Adding ones weights to compute the bias (represented as extra input)
        '''
        temp = X.tolist()
        temp.append(1)
        X_with_ones = np.array(temp)
        return X_with_ones

    def get_weights(self):
        return self.weights
