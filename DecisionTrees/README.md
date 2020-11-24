[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<p align="center">
<h1 align="center">Decision Trees</h1>
</p>


## About the Project
Developed functions:
1. [DT_train_binary](#dt_train_binary)
2. [DT_test_binary](#dt_test_binary)
3. [DT_make_prediction](#dt_make_prediction)
4. [DT_train_real](#dt_train_real)
5. [DT_test_real](#dt_test_real)

There is also a test script called [test_script.py](#test_script) which can be used to check the working of the functions


<!-- DT TRAIN BINARY -->
## DT_TRAIN_BINARY

```python
DT = DT_train_binary(X, Y, max_depth)
```

This function takes the training data **X** as input and the labels **Y**. Here the features are binary, but the feature vectors can be of any finite dimension. The training feature data **X** should be structured as a 2D numpy array, with each row corresponding to a single sample. The training labels **Y** should be structured as a 1D numpy array, with each element corresponding to a single label. **Y** should have the same number of elements as **X** has rows. **max_depth** is an integer that indicates the maximum depth for the resulting decision tree.
**DT_train_binary(X,Y,max depth)** returns the decision tree generated using information gain, limited by some given maximum depth. If max depth is set to -1 then learning only stops when we run out of features or our information gain is 0. The decision tree is returned as an object.


<!-- DT TEST BINARY -->
## DT_TEST_BINARY

```python
accuracy = DT_test_binary(X, Y, DT)
```

This function takes the test data **X** and test labels **Y** and a learned decision tree model **DT**, which is returned from the **DT_train_binary** function, and returns the accuracy (from 0 to 1) on the test data using the decision tree for predictions.


<!-- DT MAKE PREDICTION -->
## DT_MAKE_PREDICTION

```python
prediction = DT_make_prediction(x, DT)
```

This function take a single sample **x** and a trained decision tree **DT** and return a single classification.


<!-- DT TRAIN REAL -->
## DT_TRAIN_REAL

```python
DT = DT_train_real(X, Y, max_depth)
```
This function takes the training data **X** as input and the labels **Y**. Here the features are real, but the feature vectors can be of any finite dimension. The training feature data **X** should be structured as a 2D numpy array, with each row corresponding to a single sample. The training labels **Y** should be structured as a 1D numpy array, with each element corresponding to a single label. **Y** should have the same number of elements as **X** has rows. **max_depth** is an integer that indicates the maximum depth for the resulting decision tree.
**DT_train_real(X,Y,max depth)** returns the decision tree generated using information gain, limited by some given maximum depth. If max depth is set to -1 then learning only stops when we run out of features or our information gain is 0. The decision tree is returned as an object.


<!-- DT TEST REAL -->
## DT_TEST_REAL

```python
accuracy = DT_test_real
```
This function takes the test data **X** and test labels **Y** and a learned decision tree model **DT**, which is returned from the **DT_train_real** function, and returns the accuracy (from 0 to 1) on the test data using the decision tree for predictions.


## Contact

### Akshay Krishna

-  Website: [Akshay Krishna - Website](https://about.me/akrishna/)
-  LinkedIN: [akshay-krishna](https://www.linkedin.com/in/akshay-krishna-ak)
-  Email: [akshay.krishna154@gmail.com](mailto:akshay.krishna154@gmail.com)

[contributors-shield]: https://img.shields.io/github/contributors/saberzuko/MachineLearningAlgorithms.svg?style=flat-square
[contributors-url]: https://github.com/saberzuko/MachineLearningAlgorithms/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/saberzuko/MachineLearningAlgorithms.svg?style=flat-square
[forks-url]: https://github.com/saberzuko/MachineLearningAlgorithms/network/members
[stars-shield]: https://img.shields.io/github/stars/saberzuko/MachineLearningAlgorithms.svg?style=flat-square
[stars-url]: https://github.com/saberzuko/MachineLearningAlgorithms/stargazers
[issues-shield]: https://img.shields.io/github/issues/saberzuko/MachineLearningAlgorithms.svg?style=flat-square
[issues-url]: https://github.com/saberzuko/MachineLearningAlgorithms/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/akshay-krishna-ak/
[test_script]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/DecisionTrees/test_script.py