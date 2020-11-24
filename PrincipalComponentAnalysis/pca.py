import numpy as np

def compute_Z(X, centering=True, scaling=False):
    # This function is used to center each of the feature with its mean
    # It is also used to standardize the features based on the inputs
    # centering and scaling
    X = np.array(X, dtype=np.float64)
    if centering:
        # if centering is true, then we subtract each feature with the respective feature mean
        means = np.mean(X, axis=0, dtype=np.float64)
        for i in range(X.shape[1]):
            X[:,i] -= means[i]
    
    if scaling:
        # if scaling is true, then we standardize the feature,
        # which is achieved by dividing each of the feature with their
        # standard deviation
        std = np.std(X, axis=0, dtype=np.float64)
        for i in range(X.shape[1]):
            X[:,i] /= std[i]

    return X

def compute_covariance_matrix(Z):
    # This function is used to compute the co-variance matrix
    # This is achieved by the dot product of transpose(Z).Z
    COV = np.dot(Z.T, Z)
    return COV

def find_pcs(COV):
    # This function is used to find the principal components
    # This function returns the principal components based
    # on the descending order of the respective eigen values
    
    # The linalg.eig function returns the eigen values L and the eigen vectors PSC
    # The ith column of the PSC is the ith eigen vector coreesponding to the ith eigen value
    L, PSC = np.linalg.eig(COV)

    # if a=[1,2,3] and b=[[1,2,3], [4,5,6], [7,8,9]]
    # then list(zip(a,b)) = [(1, [1,2,3]), (2, [4,5,6]), (3, [7,8,9])]
    zipped_list = zip(L, PSC.T)
    
    # Here we are sorting the eigen values and the eigen vectors in the decreasing order
    zipped_list = sorted(zipped_list, key = lambda item:item[0], reverse=True)
    
    # We are unzipping the zipped list
    L, PSC = list(zip(*zipped_list))[:]
    
    # Converting the elements back to matrix format
    L = np.array(L, dtype=np.float64)
    PSC = np.array(PSC).T
    return (L, PSC)

def project_data(Z, PCS, L, k, var):
    # This function is used to project the Z matrix on the eigen vectors
    # If the value of k is zero, then we use the cummulative variance to 
    # determine the value of k.
    if k == 0:
        L = np.array(L, dtype=np.float64)
        L_sum = np.sum(L)
        cummulative_var = 0
        for i in range(PCS.shape[1]):
            cummulative_var += (L[i]/L_sum)
            if cummulative_var > var:
                k = i+1
                break

    PCS = PCS[:,:k]
    Z_star = np.dot(Z, PCS)
    return (Z_star)

if __name__ == "__main__":
    X = np.array([[-1, -1],
                 [-1, 1],
                 [1, -1],
                 [1, 1]])
    Z = compute_Z(X)
    COV = compute_covariance_matrix(Z) 
    L, PCS = find_pcs(COV) 
    Z_star = project_data(Z, PCS, L, 1, 0)
    print(L)
    print(PCS)
    print(Z_star)