import numpy as np
import helpers
import binary_classification as bc
import multiclass_classification as mc

print(" ")
print("SVM Binary Classification Test 1:")
data = helpers.generate_training_data_binary(1)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)

print(" ")
print("SVM Binary Classification Test 2:")
data = helpers.generate_training_data_binary(2)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)

print(" ")
print("SVM Binary Classification Test 3:")
data = helpers.generate_training_data_binary(3)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)

print(" ")
print("SVM Binary Classification Test 4:")
data = helpers.generate_training_data_binary(4)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)

print(" ")
print("SVM Multi-Class Classification Test:")
[data,Y] = helpers.generate_training_data_multi(1)
[W,B] = mc.svm_train_multiclass([data,Y])
print(W,B)