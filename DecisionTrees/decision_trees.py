import numpy as np
import math

def counter(rows):
    # Function to count the number of zeros and number of ones
    # The output is a dictinoary containing the counts
    Nzeros = 0; Nones = 0
    for row in rows:
        label = row[-1]
        if label == 0:
            Nzeros += 1
        else:
            Nones += 1
    return {"0": Nzeros, "1":Nones}

class Question:
    """This class is used to partition the dataset
    This class records the column number (e.g. 0 for 0th feature) and a column
    value (e.g. 0 or 1) as the features are binary features. The match method is 
    used to compare the feature value stored in the question.
    e.g. Question = question(1, 0) The question: is the feature_1 == 0
    question.match([1,1,1,1,1]), returns False as the array[1] is 1.
    feature_0 is array[0], feature_1 is array[1] and so on. array[-1] is label"""

    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def match(self, example):
        val = example[self.column]
        return val == self.value
    
    def __repr__(self):
        # This is a function for the purpose of printing the object
        condition = "=="
        return ("Is feature_{} {} {}?".format(str(self.column), condition, str(self.value)))

class Question_Real:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def match(self, example):
        val = example[self.column]
        return val >= self.value
    
    def __repr__(self):
        condition = ">="
        return ("Is feature_{} {} {}?".format(str(self.column), condition, str(self.value)))

def partition(rows, question):
    # For each row in the dataset, check if it matches the question
    # If it matches add to 'true rows', else add to 'false rows'
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def entropy(rows):
    # Function to calculate entropy
    # returns the Entropy value
    counts = counter(rows)
    Nzeros = counts["0"]
    Nones = counts["1"]
    if Nzeros == 0 or Nones == 0:
        return 0
    total = Nzeros + Nones
    Pzeros = Nzeros/total; Pones = 1 - Pzeros
    H = ((-Pzeros*math.log2(Pzeros))-((Pones*math.log2(Pones))))
    return H

def informationGain(left_branch, right_branch, parent_entropy):
    # Funtion to calculate the information gain
    p = float(len(left_branch)) / (float(len(left_branch)) + float(len(right_branch)))
    IG = parent_entropy - ((p * entropy(left_branch)) + ((1-p) * entropy(right_branch)))
    return IG

def best_split(rows):
    """ This function is used to find the best split by calculating the
    information gain by traversing through each feature it returns the 
    best feature and the best gain"""
    best_gain = 0
    best_question = None
    current_entropy = entropy(rows)
    n_features = len(rows[0]) - 1
    for col in range(n_features):
        # The question is, is feature_col == 0 ?
        question = Question(col, 0)
        #if the rows match with the question they are stored in true_rows else false_rows
        true_rows, false_rows = partition(rows, question)
        if len(true_rows) == 0 or len(false_rows) == 0:
            continue
        gain = informationGain(true_rows, false_rows, current_entropy)
        if gain >= best_gain:
            best_gain, best_question = gain, question
    return best_gain, best_question

def best_split_real(rows):
    best_gain = 0
    best_question = None
    current_entropy = entropy(rows)
    n_features = len(rows[0]) - 1
    for col in range(n_features):
        # Findingunique values in each column
        values = set([row[col] for row in rows])
        for val in values:
            # The question is, is feature_col >= val ?
            question = Question_Real(col, val)
            #if the rows match with the question they are stored in true_rows else false_rows
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = informationGain(true_rows, false_rows, current_entropy)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    """This class is used to hold the leaf node which is used to make the decisions.
    This class holds a dictionary which holds the number of times the class occured
    in the data"""
    def __init__(self, rows):
        self.predictions = counter(rows)

class Decision_Node:
    """This class is used to form nodes and make decisions to make
    a split. It also creates two child nodes"""
    def __init__(self,
                question,
                true_branch,
                false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class BuildTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def build_tree(self, rows, depth=0):
        """This function is used to build the tree. It uses the recurrsion method
        to build the tree. The tree stops if the gain is zero or the tree reaches the 
        max_depth"""
        if self.max_depth == -1 or depth <= self.max_depth:
            gain, question = best_split(rows)
            if gain == 0 or depth==self.max_depth:
                return Leaf(rows)
            true_rows, false_rows = partition(rows, question)
            true_branch = self.build_tree(true_rows, depth+1)
            false_branch = self.build_tree(false_rows, depth+1)
        return Decision_Node(question, true_branch, false_branch)
    
    def build_tree_real(self, rows, depth=0):
        if self.max_depth == -1 or depth <= self.max_depth:
            gain, question = best_split_real(rows)
            if gain == 0 or depth==self.max_depth:
                return Leaf(rows)
            true_rows, false_rows = partition(rows, question)
            true_branch = self.build_tree_real(true_rows, depth+1)
            false_branch = self.build_tree_real(false_rows, depth+1)
        return Decision_Node(question, true_branch, false_branch)

    def print_tree(self, node, spacing=""):
        # This function is used to print the decision tree in recursive fashion
        if isinstance(node, Leaf):
            print(spacing + "Predict", node.predictions)
            return
        print (spacing + str(node.question))
        print(spacing+ "--> True:")
        self.print_tree(node.true_branch, spacing + "   ")
        print(spacing+ "--> False:")
        self.print_tree(node.false_branch, spacing + "   ")

def test_tree(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return test_tree(row, node.true_branch)
    else:
        return test_tree(row, node.false_branch)

def print_leaf(count_dict):
    max_count = 0; max_lbl = None
    for lbl in count_dict.keys():
        if count_dict[lbl] >= max_count:
            max_count = count_dict[lbl]
            max_lbl = float(int(lbl))
    return max_lbl

def DT_train_binary(X, Y, max_depth):
    Y = Y.reshape((Y.size,1))
    training_data = np.append(X,Y,1)
    del X; del Y
    tree = BuildTree(max_depth)
    decision_tree = tree.build_tree(training_data)
    # The following commented lines can be used to print the trained decision tree
    # print("The trained Decision Tree")
    # tree.print_tree(decision_tree)
    # print("Choose the max count values to make the prediction")
    # print()
    return(decision_tree)

def DT_test_binary(X, Y, DT):
    Y = Y.reshape((Y.size,1))
    testing_data = np.append(X,Y,1)
    del X; del Y
    count = 0; total = 0
    for row in testing_data:
        predicted = print_leaf(test_tree(row, DT))
        # To print the actual and the predicted values
        # print("Actual:  {}, Predicted: {}".format(row[-1], predicted))
        total += 1
        if row[-1] == predicted:
            count += 1
    accuracy = count/total
    return accuracy

def DT_make_prediction(x, DT):
    predicted = print_leaf(test_tree(x, DT))
    return predicted

def DT_train_real(X, Y, max_depth):
    Y = Y.reshape((Y.size,1))
    training_data = np.append(X,Y,1)
    del X; del Y
    tree = BuildTree(max_depth)
    decision_tree = tree.build_tree_real(training_data)
    # The following commented lines can be used to print the trained decision tree
    # print("The trained Decision Tree")
    # tree.print_tree(decision_tree)
    # print("Choose the max count values to make the prediction")
    # print()
    return(decision_tree)

def DT_test_real(X, Y, DT):
    Y = Y.reshape((Y.size,1))
    testing_data = np.append(X,Y,1)
    del X; del Y
    count = 0; total = 0
    for row in testing_data:
        predicted = print_leaf(test_tree(row, DT))
        # To print the actual and the predicted values
        # print("Actual:  {}, Predicted: {}".format(row[-1], predicted))
        total += 1
        if row[-1] == predicted:
            count += 1
    accuracy = count/total
    return accuracy