import numpy as np 






X = np.array([[1,2,3,4], [4,3,5,3], [1,2,3,3]])
X_shape = X.shape
print(X_shape)
o =np.ones((X_shape[0],X_shape[1]+1))
o[:, 0:X_shape[1]] = X
print(o)




































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
