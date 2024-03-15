import numpy as np 
from math import e


identity = lambda y : y

ReLU = lambda y : y if y >= 0 else 0

leaky_ReLU = lambda y : y if y >= 0 else 0.1*y

Sigmoid = lambda y : 1 / (1 + e ** -y)

Signum = lambda y : 1 if y >= 0 else -1

Tanh = lambda y, a : (1 - e**(-y*a)) / (1 + e**(-y*a))   


d_identity = lambda y : 1

d_ReLU = lambda y : 1 if y >= 0 else 0

d_leaky_Relu = lambda y : 1 if y >= 0 else 0.1

d_Sigmoid = lambda y : Sigmoid(y) * (1 - Sigmoid(y))

d_Signum = lambda y : 0

d_Tanh = lambda y, a: 2*a*((e**(-y*a)) / (1 + e**(-y*a))**2)
