from scipy.spatial import distance
import numpy as np

def signum(num):
    # FUnction to identify the sign of the number
    if num >= 0:
        return 1.0
    elif num < 0:
        return -1.0

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    """This function returns the accuracy of the predictions made
    given the training data, testing data, and K """
    Y_train = Y_train.reshape((Y_train.size, 1))
    train_data = np.append(X_train, Y_train, 1)
    del X_train; del Y_train
    
    Y_test = Y_test.reshape((Y_test.size, 1))
    test_data = np.append(X_test, Y_test, 1)
    del X_test; del Y_test
    
    """The count variable stores the number of times correct predictions were made
    and is used to calculate the accuracy."""
    count = 0; total = 0
    for test_idx in range(len(test_data)):
        norm_list = []
        for train_idx in range(len(train_data)):
            norm = distance.euclidean(test_data[test_idx][:-1], train_data[train_idx][:-1])
            norm_list.append([norm, train_data[train_idx][-1]])
        norm_list.sort(key=lambda x:x[0])
        norm_list = norm_list[:K]
        vote = sum([lbl[-1] for lbl in norm_list])
        predicted = signum(vote)
        # print("Actual: {}, Predicted: {}".format(test_data[test_idx][-1], predicted))
        if predicted == test_data[test_idx][-1]:
            count += 1
        total += 1
    accuracy = count/total
    return accuracy

def choose_K(X_train, Y_train, X_val, Y_val):
    """Using the KNN_test method for returning the accuracy and chosing the best accuracy
    while iterating through the values of K and chosing the K which has the best accuracy"""
    K_list = [k for k in range(len(X_train))]
    best_accuracy = 0; best_K = None
    for K in K_list:
        accuracy = KNN_test(X_train, Y_train, X_val, Y_val, K)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_K = K
    return best_K 