# The functions used in binary_classification.py are used here with slight modifications
# to accomadate predictions of multiple class
import numpy as np

def distance(a, b):
    return((np.sum((a - b)**2))**0.5)

def L2norm(vec):
    vec = np.array(vec)
    return((np.sum(vec**2))**0.5)

def distance_point_to_hyperplane(pt, w, b):
    pt = np.array(pt)
    w = np.array(w)
    try:
        dist = abs(np.dot(pt, w) + b)/L2norm(w)
        return(dist)
    except:
        return(float("inf"))

def compute_margin(data, w, b, pos_class, *argv):
    # as we are using the methodology of 1 v/s not 1, we are sending the
    # label information as well
    min_dist = float("inf")
    data = np.array(data)
    w = np.array(w)
    for sample in data:
        if sample[-1] == pos_class:
            mul = 1
        else:
            mul = -1
        pred = mul*(np.dot(sample[:-1], w) + b)
        if pred < 0:
            return (-float("inf"))
        distance = distance_point_to_hyperplane(sample[:-1], w, b)
        if distance < min_dist:
            min_dist = distance
    for support_vector in argv:
        support_vector_distance = distance_point_to_hyperplane(support_vector, w, b)
        if round(support_vector_distance, 4) != round(min_dist, 4):
            return (-float("inf"))
    return(min_dist)
            

def svm_train_brute(positive_samples, negative_samples, positive_sample_class):
    training_data = np.append(positive_samples, negative_samples, axis=0)
    max_margin2 = -float("inf")
    max_margin3 = -float("inf")

    for pos in positive_samples:
        for neg in negative_samples:
            pos_sample = pos[:-1]
            neg_sample = neg[:-1]
            dirW = pos_sample - neg_sample
            W = dirW/L2norm(dirW)
            possible_margin = distance(pos_sample, neg_sample)/2
            W = W/possible_margin
            b = 1 - np.dot(W, pos_sample)
            margin = compute_margin(training_data, W, b, positive_sample_class, pos_sample, neg_sample)
            if (round(margin, 4) > round(max_margin2, 4)):
                max_margin2 = margin
                final_w2 = W.copy()
                final_b2 = b
    
    for index1 in range(len(training_data)):
        for index2 in range(len(training_data)):
            for index3 in range(len(training_data)):
                if (index1 != index2 and index2 != index3 and index3 != index1):
                    coeffs1 = np.append(training_data[index1][:-1], 1)
                    coeffs2 = np.append(training_data[index2][:-1], 1)
                    coeffs3 = np.append(training_data[index3][:-1], 1)
                    sol1 = 1 if(training_data[index1][-1] == positive_sample_class) else -1
                    sol2 = 1 if(training_data[index2][-1] == positive_sample_class) else -1
                    sol3 = 1 if(training_data[index3][-1] == positive_sample_class) else -1
                    sols = np.array([sol1, sol2, sol3])
                    a = np.array([coeffs1, coeffs2, coeffs3])
                    try:
                        W = np.linalg.solve(a, sols)
                        if np.all(W[:2] == np.array([0,0])):
                            continue
                        margin = compute_margin(training_data, W[:2], W[-1], positive_sample_class, training_data[index1][:-1], training_data[index2][:-1], training_data[index3][:-1])
                    except:
                        continue
                    if (round(margin, 4) > round(max_margin3, 4)):
                        max_margin3 = margin
                        final_w3 = W[:2]
                        final_b3 = W[-1]
    if round(max_margin2, 4) > round(max_margin3, 4):
        return(final_w2, final_b2)
    else:
        return(final_w3, final_b3)

def svm_train_multiclass(training_data):
    data = training_data[0]
    num_classes = training_data[1]
    W = list()
    B = list()
    for i in range(1, num_classes+1):
        yes_class = data[data[:,-1] == i]
        no_class = data[data[:, -1] != i]
        w, b = svm_train_brute(yes_class, no_class, i)
        W.append(w)
        B.append(b)
    W = np.array(W)
    B = np.array(B)
    return(W, B)

def svm_test_multiclass(W, B, x):
    predictions = list()
    max_dist = -float("inf")
    for index in range(len(B)):
        pred = round(np.dot(W[index], x) + B[index], 4)
        if pred >= 1:
            predictions.append(1)
        elif pred <= -1:
            predictions.append(0)
    predictions = np.array(predictions)
    if np.sum(predictions) == 0:
        return -1
    elif np.sum(predictions) > 1:
        indices = np.where(predictions == 1)[0]
        for i in indices:
            d = distance_point_to_hyperplane(x, W[i], B[i])
            if d >= max_dist:
                max_dist = d
                predicted_class = i+1
        return (predicted_class)
    else:
        index = np.where(predictions == 1)[0]
        return (index + 1)
