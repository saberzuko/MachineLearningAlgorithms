import numpy as np
import helpers
import pca

print("PCA Test 1:")
X,Y = helpers.load_data("data_1.txt")
Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z) 
L, PCS = pca.find_pcs(COV) 
Z_star = pca.project_data(Z, PCS, L, 1, 0)
print(L)
print(PCS)
print(Z_star)

print(" ")
print("PCA Test 2:")
X,Y = helpers.load_data("data_2.txt")
Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z) 
L, PCS = pca.find_pcs(COV) 
Z_star = pca.project_data(Z, PCS, L, 1, 0)
print(L)
print(PCS)
print(Z_star)