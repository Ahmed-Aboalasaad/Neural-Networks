import numpy as np 


e = 2.7182818

# activation functions
identity = lambda y : y

ReLU = lambda y : y if y >= 0 else 0

leaky_ReLU = lambda y : y if y >= 0 else 0.1*y

Sigmoid = lambda y : 1 / (1 + e ** -y)

Signum = lambda y : 1 if y >= 0 else -1

Tanh = lambda y : (1 - e**(-y)) / (1 + e**(-y))   



# derivatives of activation functions
d_identity = lambda y : 1

d_ReLU = lambda y : 1 if y >= 0 else 0

d_leaky_Relu = lambda y : 1 if y >= 0 else 0.1

d_Sigmoid = lambda y : Sigmoid(y) * (1 - Sigmoid(y))

d_Signum = lambda y : 0

d_Tanh = lambda y: 2*((e**(-y)) / (1 + e**(-y))**2)