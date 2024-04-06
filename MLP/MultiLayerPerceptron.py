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
        self.__init_weights()  # initializing weights between ALL layers



    def fit(self, X: np.array, y: np.array):
        self.examples_number = X.shape[0]
        # self.weights = [np.array([[0.21, -0.4], [0.15, 0.1], [-0.3, 0.24]]), np.array([[-0.2], [0.3], [-0.4]])]
        if self.add_bias:
            X = self.__add_ones(X)
        
        for j in range(self.epochs):
            net_inputs = []  # List of lists: each of which has the net input of a layer
            input = X

            ### Feeding Forward
            for cluster_index in range(self.hidden_layers_number + 1):
                # calculate activations 
                net_input = input.dot(self.weights[cluster_index]) # input [dot product] Weights
                net_inputs.append(net_input)
                activation = self.activation_function(net_input)

                # The activation in the current iteration will be the input for the next one
                input = activation
                # if the user asked for bias & it's not the last activation
                if self.add_bias and cluster_index != self.hidden_layers_number:
                    input = self.__add_ones(input)

            
            ### Back Probagation            
            gradients = [np.zeros((1, i)) for i in self.neurons_per_layer]
            gradients.append(np.zeros((1, self.output_size)))

            # Cost gradients with respect to output-layer neurons
            output_layer_error = y - input  # at this point, input variable holds the actiavation of the output neurons
            gradients[self.hidden_layers_number][:] = ((output_layer_error/self.examples_number) * self.activation_function_derivative(net_inputs[self.hidden_layers_number])).sum(axis=0)

            # Cost gradients with respect to hidden layers neurons
            for cluster_index in range(self.hidden_layers_number - 1, -1, -1):
                gn_t_w = gradients[cluster_index+1].dot(self.weights[cluster_index+1][:-1].T)
                gradients[cluster_index][:] = gn_t_w * self.activation_function_derivative(net_inputs[cluster_index]).sum(axis=0)


            ### Updating Weights
            # I (as a non-input neuron) update the weights that get into me using the cost gradient with respect ot me
            # weight update equation: new = old + learning_rage * the cost gradient with respect ot me * input value through this weight
            
            previous_activation = X
            # iterate over weight clusters
            for cluster_index, weight_cluster in enumerate(self.weights):
                # Iterate over neurons in the go-to layer of this weights cluster
                for neuron_index in range(weight_cluster.shape[1]):
                    weight_cluster = weight_cluster.T
                    
                    weight_cluster[neuron_index][:] += self.learning_rate * gradients[0][neuron_index] * previous_activation

                    

                    weight_cluster = weight_cluster.T
                    self.weights[cluster_index] = weight_cluster
                previous_activation = 



    def predict(self, x):
        pass



    def __init_weights(self):
        '''
        Initializes the weights list with np arrays filled with a proper number of random weights each
        '''
        weights = []

        # [2, 4, 1] means 2 neurons in input layer, 4 neurons in a single hidden layer, and 1 output neuron in the output layer
        neurons_count = [self.input_size]
        neurons_count.extend(self.neurons_per_layer)
        neurons_count.append(self.output_size)
        shortcut = 1 if self.add_bias else 0

        for i in range(self.hidden_layers_number + 1):
            current_next_weights = np.random.rand(neurons_count[i] + shortcut, neurons_count[i+1])
            weights.append(current_next_weights)
        self.weights = weights    

    def __add_ones(self, X):
        '''
        Adding ones weights to the weights matrix as if the bias is an extra input
        '''
        X_shape = X.shape
        X_with_ones =np.ones((X_shape[0],X_shape[1]+1))
        X_with_ones[:, 0:X_shape[1]] = X
        return X_with_ones

    def get_weights(self):
        return self.weights
