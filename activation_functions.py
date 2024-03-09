EULER_CONSTANT = 2.7182818

identity = lambda y : y

ReLU = lambda y : y if y >= 0 else 0

leaky_ReLU = lambda y : y if y >= 0 else 0.1*y

sigmoid = lambda y : 1 / (1 + EULER_CONSTANT ** -y)

signum = lambda y : 1 if y >= 0 else 0

tanh = lambda y, a : (1 - EULER_CONSTANT**(-y*a)) / (1 + EULER_CONSTANT**(-y*a))   
