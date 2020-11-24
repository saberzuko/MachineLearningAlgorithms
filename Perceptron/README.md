[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<p align="center">
<h1 align="center">Perceptron</h1>
</p>


## About the Project
Developed functions:
1. [pereptron_train](#perceptron_train)
2. [perceptron_test](#perceptron_test)

There is also a test script called [test_script.py][test_script] which can be used to check the working of the functions


## PERCEPTRON_TRAIN

```python
(w, b) = perceptron_train(X, Y)
```

This function takes the takes training data **X**, **Y** as input and outputs the weights **w**, and the bias **b** of the perceptron. The function can handle any real-valued features, with feature vectors in any dimension, and binary labels.


## PERCEPTRON_TEST

```python
accuracy = perceptron_test(X_test, Y_test, w, b)
```

This function takes the testing data **X_test**, **Y_test**, the perceptron weights **w** and the bias **b** as input and returns the accuracy on the testing data.


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
[test_script]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/Perceptron/test_script.py