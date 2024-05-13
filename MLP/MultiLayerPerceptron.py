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
                 input_size : int, # number of neurons in the input layer
                 output_size : int,  # number of neurons in the output layer
                 hidden_layers_number : int, 
                 neurons_per_layer : list,  # number of neurons in each hidden layer
                 learning_rate : float, 
                 activation_function, # a string that get parsed to a lambda expression
                 epochs : int,
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
        self.weights = self.__init_weights()  # initializing random weights between all layers
        print(f'\n\nInitial Weights:\n{self.weights}')

    def fit(self, X: np.array, y: np.array):
        self.data_size = X.shape[0]
        # self.weights = [np.array([[0.21, -0.4], [0.15, 0.1], [-0.3, 0.24]]), np.array([[-0.2], [0.3], [-0.4]])]
        ## append 1 at the end of each record if add_bias is true
        if self.add_bias:
            X = self.__add_ones(X)
        
        # List of lists: each of which has the net outputs of a layer
        for j in range(self.epochs):
            for record, label in zip(X, y):    
                outputs = []
                current_input = record
                activations = [record] 
                ### Feeding Forward [calculating output]
                for i in range(self.hidden_layers_number + 1):
                    output = current_input.dot(self.weights[i]) # input * Weights 
                    outputs.append(output)

                    activation = self.activation_function(output)
                    current_input = activation

                    # if it's not the output layer & the user asked for bias
                    
                    if i != self.hidden_layers_number and self.add_bias:
                        current_input = self.__add_ones2(current_input)
                    activations.append(current_input)
                
                ### backward step [calculating cost gradients with respect to neurons]
                output_layer_error = label - current_input  # error calculated in output layer
                # initializing gradients array
                gradients = [np.zeros((1, n_neurons)) for n_neurons in self.neurons_per_layer]
                gradients.append(np.zeros((1, self.output_size)))
                
                # Output-layer gradient
                gradients[self.hidden_layers_number][:] = ((output_layer_error) * self.activation_function_derivative(outputs[self.hidden_layers_number])).mean(axis=0)
                # Calculate cost gradients with respect to neurons in hidden layers
                for i in range(self.hidden_layers_number - 1, -1, -1):
                    if self.add_bias:
                        gn_t_w = gradients[i+1].dot(self.weights[i+1][:-1].T)
                    else :
                        gn_t_w = gradients[i+1].dot(self.weights[i+1].T)

                    gradients[i][:] = gn_t_w * self.activation_function_derivative(outputs[i])
                ## forward step2 [updating weights]                
                temp = []
                temp.extend(self.neurons_per_layer)
                temp.append(self.output_size)
                for i in range(self.hidden_layers_number+1):
                    current_ = activations[i]
                    val = temp[i]
                    for j in range(val):
                        self.weights[i][:,j] = self.weights[i][:,j] + self.learning_rate * gradients[i][:,j] * current_
                        
    def predict(self, X):
        if self.add_bias:
            X = self.__add_ones(X)
        local_current = X

        for i in range(self.hidden_layers_number+1):
            net = local_current.dot(self.weights[i])
            activation = self.activation_function(net)
            local_current = activation
            if i != self.hidden_layers_number and self.add_bias:
                local_current = self.__add_ones(local_current)
        return local_current

    def __init_weights(self):
        '''
        Initializes the weights list with np arrays filled with a proper number of random weights each.
        Each np array of them has the weitghts between 2 of the layers.
        '''
        weights = []

        # [2, 4, 1] means 2 neurons in input layer, 4 neurons in the single hidden layer, and 1 output neuron in the output layer
        neurons_count = [self.input_size]
        neurons_count.extend(self.neurons_per_layer)
        neurons_count.append(self.output_size)
        shortcut = 1 if self.add_bias else 0

        for i in range(self.hidden_layers_number+1):
            weights_bulk = np.random.rand(neurons_count[i] + shortcut, neurons_count[i+1])
            weights.append(weights_bulk)

        return weights
    
    def __add_ones(self, X):
        '''
        Adding ones weights to compute the bias (represented as extra input)
        '''
        X_shape = X.shape
        X_with_ones =np.ones((X_shape[0],X_shape[1]+1))
        X_with_ones[:, 0:X_shape[1]] = X
        return X_with_ones
    
    def __add_ones2(self, X):
        '''
        Adding ones weights to compute the bias (represented as extra input)
        '''
        temp = X.tolist()
        temp.append(1)
        X_with_ones = np.array(temp)
        return X_with_ones

    def get_weights(self):
        return self.weights
