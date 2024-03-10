import numpy as np 
from math import e


identity = lambda y : y

ReLU = lambda y : y if y >= 0 else 0

leaky_ReLU = lambda y : y if y >= 0 else 0.1*y

sigmoid = lambda y : 1 / (1 + e ** -y)

signum = lambda y : 1 if y >= 0 else -1

tanh = lambda y, a : (1 - e**(-y*a)) / (1 + e**(-y*a))   


d_identity = lambda y : 1

d_ReLU = lambda y : 1 if y >= 0 else 0

d_leaky_Relu = lambda y : 1 if y >= 0 else 0.1

d_sigmoid = lambda y : sigmoid(y) * (1 - sigmoid(y))

d_signum = lambda y : 0

d_tanh = lambda y, a: 2*a*((e**(-y*a)) / (1 + e**(-y*a))**2)
