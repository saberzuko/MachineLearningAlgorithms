import numpy as np

def distance(a, b):
    # This function is used to calculate the eucledian
    # distance between two points a and b
    return((np.sum((a - b)**2))**0.5)

def L2norm(vec):
    # This function is used to calculate the L2 norm of the vector vec
    vec = np.array(vec)
    return((np.sum(vec**2))**0.5)

def distance_point_to_hyperplane(pt, w, b):
    # This function is used to calculate the distnce of point pt
    # from the hyperplane which is described by w and b
    pt = np.array(pt)
    w = np.array(w)
    try:
        dist = abs(np.dot(pt, w) + b)/L2norm(w)
        return(dist)
    except:
        return(float("inf"))

def compute_margin(data, w, b, *argv):
    # This function is used to calculate the margin and is used to verify
    # if the vectors assumed as support vectors ar actual support vectors or not
    min_dist = float("inf")
    data = np.array(data)
    w = np.array(w)
    for sample in data:
        pred = sample[-1]*(np.dot(sample[:-1], w) + b)
        # if the assumed hyperplane mis classifies even a single point then
        # then the assumed hyperplane doesn't maximze the margin
        if pred < 0:
            return (-float("inf"))
        distance = distance_point_to_hyperplane(sample[:-1], w, b)
        # The if condition finds the least distance between the hyperplane and the set of points
        if distance < min_dist:
            min_dist = distance
    for support_vector in argv:
        # This for loop verifies if the assumed support vector are the actual support vector
        # by comparing the least distance and the distance between the support vector and the hyperplane
        # if all the support vectors have the same minimum distance then the assumed support vectors are
        # the actual support vectors
        support_vector_distance = distance_point_to_hyperplane(support_vector, w, b)
        if round(support_vector_distance, 4) != round(min_dist, 4):
            return (-float("inf"))
    # if this statment is reached then it means that the assumed support vectors are the actual
    # support vectors
    return(min_dist)
            

def svm_train_brute(training_data):
    training_data = np.array(training_data)
    # seggragting the positive and the negative sampes
    positive_samples = training_data[training_data[:,-1] > 0]
    negative_samples = training_data[training_data[:, -1] < 0]
    max_margin2 = -float("inf")
    max_margin3 = -float("inf")

    # iterating through each of the positive and the negative samples and computing
    # the hyperplane and b.
    for pos in positive_samples:
        for neg in negative_samples:
            pos_sample = pos[:-1]
            neg_sample = neg[:-1]
            dirW = pos_sample - neg_sample
            W = dirW/L2norm(dirW)
            possible_margin = distance(pos_sample, neg_sample)/2
            # possible hyperplane with W 
            W = W/possible_margin
            b = 1 - np.dot(W, pos_sample)
            # Computing the margin by calling the compute_margin
            margin = compute_margin(training_data, W, b, pos_sample, neg_sample)
            if (round(margin, 4) > round(max_margin2, 4)):
                # identifying the hyperplane which maximizes the margin
                max_margin2 = margin
                pos_support_vector = pos.copy()
                neg_support_vector = neg.copy()
                final_w2 = W.copy()
                final_b2 = b
                S2 = np.array([pos_support_vector, neg_support_vector])
    
    # iterating through 3 samples at a time to compute the hyperplane
    for index1 in range(len(training_data)):
        for index2 in range(len(training_data)):
            for index3 in range(len(training_data)):
                if (index1 != index2 and index2 != index3 and index3 != index1):
                    coeffs1 = np.append(training_data[index1][:-1], 1)
                    coeffs2 = np.append(training_data[index2][:-1], 1)
                    coeffs3 = np.append(training_data[index3][:-1], 1)
                    sols = np.array([training_data[index1][-1], training_data[index2][-1], training_data[index3][-1]])
                    a = np.array([coeffs1, coeffs2, coeffs3])
                    try:
                        # Solving a set of 3 equations to identify the values of w and b
                        W = np.linalg.solve(a, sols)
                        if np.all(W[:2] == np.array([0,0])):
                            continue
                        # computing the margin for a given hyperplane
                        margin = compute_margin(training_data, W[:2], W[-1], training_data[index1][:-1], training_data[index2][:-1], training_data[index3][:-1])
                    except:
                        continue
                    # verifying if the hyperplane is having the maximum margin
                    if (round(margin, 4) > round(max_margin3,4)):
                        max_margin3 = margin
                        final_w3 = W[:2]
                        final_b3 = W[-1]
                        S3 = np.array([training_data[index1], training_data[index2], training_data[index3]])
    # Comparing the hyperplane identified by 2 points and 3 points and comparing theirs margins
    # returning the w, b and support vectors of the hyperplane having the largest margin
    if round(max_margin2, 4) > round(max_margin3, 4):
        return(final_w2, final_b2, S2)
    else:
        return(final_w3, final_b3, S3)

def svm_test_brute(w, b, x):
    # using the dot product to make the predictions of the test data x
    w = np.array(w)
    x = np.array(x)
    pred = round(np.dot(x, w) + b, 4)
    if pred >= 1:
        return(1)
    elif pred <= -1:
        return(-1)