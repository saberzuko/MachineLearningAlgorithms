[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<br />
<p align="center">
<h1 align="center">K-Nearest Neighbors</h1>
</p>


## About the Project
Developed Functions
1. [KNN_test(X_train, Y_train, X_test, Y_test, K)](knn_test)
2. [choose_K(X_train, Y_train, X_val, Y_val)](choose_k)

There is also a test script called [test_script.py][test_script] which can be used to check the working of the functions

## KNN_TEST

```python
accuracy = KNN_test(X_train, Y_train, X_test, Y_test, K)
```

This function takes training data **X_train**, **Y_train**, test data **X_test**, **Y_test**, and **K** as inputs. **KNN_test(X_train, Y_train, X_test, Y_test, K)** should return the accuracy on the test data. This function can handle any finite dimension feature vector, with real-valued features. Here the labels are binary, and they should be -1 for the negative class and +1 for the positive class.


## CHOOSE_K

```python
K = choose_K(X_train, Y_train, X_val, Y_val)
```

This function takes the training data **X_train**, **Y_train** and validation data **X_val**, **Y_val** as inputs and returns a **K** value. This function iterates through all possible K values and choose the best K for the given training data and validation data. The returned **K** will be in order to achieve the best accuracy on the validation data.


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
[test_script]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/KNearestNeighbors/test_script.py