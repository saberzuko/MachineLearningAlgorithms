import numpy as np

def magnitude(array):
    return((np.sum(array * array))**0.5)

def gradient_descent(gradient_func, x_init, neeta):
    while True:
        # Computing the gradient using the given function
        grad = gradient_func(x_init)

        # if the magnitude of the gradient is less than 0.0001
        # then return the x_init which gave that particular magnitude
        if (magnitude(grad) <= 0.0001):
            return(x_init)
        
        # Performing the gradient descent calculation
        x_init -= (neeta * grad)
