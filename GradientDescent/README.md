[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<p align="center">
<h1 align="center">Gradient Descent</h1>
</p>


## About the Project
Developed functions:
1. [gradient_descent](#gradient_descent)

There is also a test script called [test_script.py][test_script] which can be used to check the working of the functions


## GRADIENT_DESCENT

```python
x = gradient_descent(deltaF, xinit, neeta)
```

This function takes the gradient of a function f, **deltaF** as input, the starting **xinit** and the learning rate **neeta**. The output of gradient descent is the value of **x** that minimizes **deltaF**. **deltaF** will be in the form of a function, so that we can calculate the gradient at a particular point. That is **deltaF** is a function with one input x and it outputs the value of the gradient at that point. If we are working with 1D variables, then x= (x1). If x is 2D then x= (x1, x2) and so on. x should be a 1D numpy array.


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
[test_script]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/GradientDescent/test_script.py