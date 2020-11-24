import numpy as np

def perceptron_train(X, Y):
    # This function is used to train a perceptron
    Y = Y.reshape((Y.size, 1))
    training_data = np.append(X, Y, 1)
    # Appending the training sample labels to the training data
    del X; del Y
    # Initializing the initial weights to zero
    W = np.zeros((len(training_data[0][:-1]),1))
    # making a copy of the weight vector to compare with the updated weights
    W_new = W.copy()
    B = 0
    B_new = 0
    epochs = 1
    while epochs <= 100:
        # We have placed the weight update rule in while loop to train 
        # for multiple epochs
        for sample in training_data:
            # Computing the y_predicted by taking the dot product
            # of the sample and the weight vector and adding the bisa
            y_pred = np.dot(sample[:-1], W_new) + B_new
            y = sample[-1]
            x = sample[:-1]
            x = x.reshape((x.size, 1))
            # if the y_pred and y do not match then we update the weights and bias
            if (y_pred * y) <= 0:
                W_new += (y*x)
                B_new += y
        # If the previous weights are same as the updated weights return the weights and the bias
        if (np.all(W == W_new) and B == B_new):
            return (W_new, B_new)
        else:
            W = W_new.copy()
            B = B_new
        epochs += 1
    # If the algorithm didn't converge for 100 epochs return the last set of weights and bias
    return (W_new, B_new)

def perceptron_test(X, Y, w, b):
    Y = Y.reshape((Y.size, 1))
    testing_data = np.append(X, Y, 1)
    total = len(testing_data)
    count = 0
    for sample in testing_data:
        y = sample[-1]
        x = sample[:-1]
        y_pred = np.dot(x, w) + b
        if (y_pred*y > 0):
            count += 1
    return (count/total)
