import numpy as np
from scipy.spatial import distance
import random

def mu_generator(X, K):
    # Function to initialize the cluster centers
    # The input is the training data X and the number of cluster centers
    mu = []; rand_keys = []
    for _ in range(K):
        rand = random.randint(0, len(X)-1)
        # The while loop prevents the random key to be repeated
        # as we want unique cluster centers
        while rand in rand_keys:
            rand = random.randint(0, len(X)-1)
        rand_keys.append(rand)
        mu.append(X[rand])
    mu = np.array(mu)
    return mu

def K_Means(X, K, mu):
    # This function is used to train our K-Means clustering algorithm and
    # return the converged cluster centers
    if len(mu) == 0:
        # If the initial clusters are not initilaized we call the mu_generator( )
        mu = mu_generator(X, K)
    clusters = {}
    # Keeping the track of the cluster centers
    updated_mu = mu.copy()

    for cluster in range(K):
        # The clusters ranges from 0 to K-1
        clusters[cluster] = []
    
    for row in X:
        least_dist = float("inf"); cluster_idx = None
        for idx in range(len(mu)):
            # Computing the eucledian distance between the sample and the cluster centers
            euclid_dist = distance.euclidean(row, mu[idx])
            # Finding the least distance between the input sample and the cluster center
            # and appending the sample to the corresponding cluster
            if euclid_dist <= least_dist:
                least_dist = euclid_dist
                cluster_idx = idx
        clusters[cluster_idx].append(row)

    for cluster in range(K):
        # if the cluster is empty then continue
        if len(clusters[cluster]) == 0:
            continue
        for dim in range(len(X[0])):
            # Computing the average of the clusters to find the new cluster centers
            avg = sum([i[dim] for i in clusters[cluster]])/len(clusters[cluster])
            updated_mu[cluster][dim] = avg
    
    if np.all(mu == updated_mu):
        # If the updated cluster centers is equal to the original cluster
        # centers stop the training process and return the cluster centers
        return updated_mu
    # else call again the K_Means( ) with the updated clusters as input
    return(K_Means(X, K, updated_mu))

def K_Means_better(X, K):
    # This funcion calls the K_Means algorithm multiple times to find the best converged
    # cluster centers
    cluster_centers = []; better_mu = {}

    for _ in range(int(len(X)/2)):
        # We use this loop to create multiple sets of cluster centers
        rand_mu = mu_generator(X, K)
        cluster_centers.append(rand_mu)

    for idx in range(len(cluster_centers)):
        # We compute the converged cluster centers for each of the cluster in cluster_centers
        mu = (K_Means(X, K, cluster_centers[idx]))
        # converting the list of lists to tuples of tuples so can use them as keys to dictionary
        tmp = tuple(tuple(i) for i in mu)
        # Computing how many times the converged cluster centers have been repeated and returning
        # the cluster center with the highest vote
        if tmp in better_mu.keys():
            better_mu[tmp] += 1
        else:
            better_mu[tmp] = 1

    cluster_centers = [(value,key) for key, value in better_mu.items()]
    final_cluster = np.array(max(cluster_centers)[1])
    return final_cluster  
