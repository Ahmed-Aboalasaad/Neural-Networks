import numpy as np 
import activation_functions


# gradients = [np.zeros((1, n_neurons)) for n_neurons in [3, 3, 2]]
# net_outputs = [np.array([5, 8, 6]), np.array([6, 4, 9]), np.array([6, 4])]
# hidden_layers_number = 2
# output_layer_error = np.array([5, 6])
# gradients[hidden_layers_number][:] = output_layer_error * np.vectorize(activation_functions.d_Sigmoid)(net_outputs[hidden_layers_number])

# print(gradients[hidden_layers_number])



lst = [1,2,3,4]
hidden_number = 3


for i in range(hidden_number-1, -1, -1):
    print(lst[i])























# def init_weights():
#     list_of_weights = []

#     neurons_per_all_layers = [input_size]
#     neurons_per_all_layers.extend(neurons_per_layer)
#     neurons_per_all_layers.append(output_size)

#     for i in range(number_of_hidden_layers+1):
#         weights_arr = np.random.random_sample((neurons_per_all_layers[i], neurons_per_all_layers[i+1]))
#         list_of_weights.append(weights_arr)

#     return list_of_weights





# epochs = 1
# x = [[1,2,3,4], [4,3,2,1], [5,6,7,8]]
# X = np.array(x)
# e = 2.7182818
# list_of_weights = []
# input_size, output_size = 4, 3
# neurons_per_layer = [4,4]
# number_of_hidden_layers = 2
# activation_function = lambda net : 1/(1+e**(-1*net))
# list_of_weights = init_weights()

# def print_weights():
#     for i in range(len(list_of_weights)):
#         print(list_of_weights[i], '\n')

# print_weights()


# print("#"*50)
#     ## append 1 at the end of each record if add_bias is true
# for i in range(epochs):    
#     current = X
#     for i in range(number_of_hidden_layers+1):
#     ## forward step 1
#         net = current.dot(list_of_weights[i])
#         y_ = np.vectorize(activation_function)(net)
#         current = y_
#         print(current, '\n')




# # ones = [1,1]
# # lst = [[1,2,3], [2,5,6]]

# # ar = np.array(lst)
