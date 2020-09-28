import numpy as np
from scipy.spatial import distance
import random

def mu_generator(X, K):
    # Function to initialize the cluster centers
    mu = []; rand_keys = []
    for _ in range(K):
        rand = random.randint(0, len(X)-1)
        while rand in rand_keys:
            rand = random.randint(0, len(X)-1)
        rand_keys.append(rand)
        mu.append(X[rand])
    mu = np.array(mu)
    return mu

def K_Means(X, K, mu):
    if len(mu) == 0:
        mu = mu_generator(X, K)
    clusters = {}
    updated_mu = mu.copy()

    for cluster in range(K):
        clusters[cluster] = []
    
    for row in X:
        least_dist = float("inf"); cluster_idx = None
        for idx in range(len(mu)):
            euclid_dist = distance.euclidean(row, mu[idx])
            if euclid_dist <= least_dist:
                least_dist = euclid_dist
                cluster_idx = idx
        clusters[cluster_idx].append(row)

    for cluster in range(K):
        if len(clusters[cluster]) == 0:
            continue
        for dim in range(len(X[0])):
            avg = sum([i[dim] for i in clusters[cluster]])/len(clusters[cluster])
            updated_mu[cluster][dim] = avg
    
    if np.all(mu == updated_mu):
        return updated_mu

    return(K_Means(X, K, updated_mu))

def K_Means_better(X, K):
    cluster_centers = []; better_mu = {}

    for _ in range(int(len(X)/2)):
        rand_mu = mu_generator(X, K)
        cluster_centers.append(rand_mu)

    for idx in range(len(cluster_centers)):
        mu = (K_Means(X, K, cluster_centers[idx]))
        tmp = tuple(tuple(i) for i in mu)
        if tmp in better_mu.keys():
            better_mu[tmp] += 1
        else:
            better_mu[tmp] = 1

    cluster_centers = [(value,key) for key, value in better_mu.items()]
    final_cluster = np.array(max(cluster_centers)[1])
    return final_cluster  