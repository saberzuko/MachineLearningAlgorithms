import numpy as np
import decision_trees as dt

X = np.array([[0,1,0,1],
            [1,1,1,1],
            [0,0,0,1]])
Y = np.array([[1],[1],[0]])
max_depth = 3

DT = dt.DT_train_binary(X,Y,max_depth)
test_acc = dt.DT_test_binary(X,Y,DT)
print("DT:",test_acc)
prediction = dt.DT_make_prediction(X[2], DT)
print("Prediction: ", prediction)