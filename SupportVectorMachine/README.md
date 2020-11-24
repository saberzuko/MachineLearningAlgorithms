[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<p align="center">
<h1 align="center">Support Vector Machine</h1>
</p>


## About the Project
Here we are implementing a linear classifier with a decision boundary that maximizes the margin between positive and negative samples (SVM). The details of SVM can be quite complex, so here we will implement a simpler version of it for 2D data.
In the [test_script.py][test_script] we make utilize of `helpers.generate_training_data_binary(num)` for generating the training data, where **num** indicates the training set. There are four different datasets (num=1, num=2, num=3, num=4) which we have provided in the [helpers.py][helpers]. All the datasets are linearly separable. **data** is a numpy array containing 2D features and their corresponding label in each row. So data[0] will have the form **[x1, x2, y]** where **[x1, x2]** is the feature for the first training sample and it has the label **y**. **y** can be -1 or 1.

Developed functions:
1. [svm_train_brute](#svm_train_brute)
2. [distance_point_to_hyperplane](#distance_point_to_hyperplane)
3. [compute_margin](#compute_margin)
4. [svm_test_brute](#svm_test_brute)
5. [svm_train_multiclass](#svm_train_multiclass)
6. [svm_test_multiclass](#svm_test_multiclass)

There is also a test script called [test_script.py][test_script] which can be used to check the working of the functions


## SVM_TRAIN_BRUTE

```python
[w, b, S] = svm_train_brute(training_data)
```

This function will return the decision boundary for our classifier **w**, **b** and a numpy array of support vectors **S**. That is, it will return the line characterized by **w** and **b** that separates the training data with the largest margin between the positive and negative class. Since the decision boundary is a line in the case of separable 2D data, we can find the best decision boundary (the one with the maximum margin between positive and negative samples) by looking at lines that separate the data and choosing the best one (the one with the maximum margin). The maximum margin separator depends only on the support vectors. So we can find the maximum margin separator via a brute force search over possible support vectors. Our method of implementation uses the brute force approach to find the hyperplane.

In order to implement our training function, we will need to write some helper functions, which are described as follows:


## DISTANCE_POINT_TO_HYPERPLANE

```python
dist = distance_point_to_hyperplane(pt, w, b)
```
This function will compute the distance between a point **pt** and a hyperplane. In 2D, the hyperplane is a line, defined by **w** and **b**. Our code will work for a hyperplanes in 2D only.


## COMPUTE_MARGIN

```python
margin = compute_margin(data, w, b)
```

This function takes a set of data **data**, and a separator **w**, **b**, and computes the margin.


## SVM_TEST_BRUTE

```python
y = svm_test_brute(w, b, x)
```

This function is used to test the new data **x** given a decision boundary **w**, **b**.


Now, let's assume we're working with data that comes from more than two classes. We can still use an SVM to classify data into one of the Y classes by using multiple SVMs in an one-vs-all fashion. Instead of thinking about it as class 1 versus class 2 versus class 3, we can think of it as class 1 versus not class 1, and class 2 versus not class 2. So we would have one binary classifier for each class to distinguish that class from the rest.
We will use `[data, Y] = generate_training_data_multi(num)` function in the [test_script.py][test_script] to generate our training data.
The following functions are used to train our multi label classifier


## SVM_TRAIN_MULTICLASS

```python
[W, B] = svm_train_multiclass(training_data)
```

This function uses our previously implemented `svm_train_brute` to train one binary classifier for each class. It will return **Y** decision boundaries, one for each class (one-vs-rest). **W** is an array of **w** and **B** is an array of **b** where w<sub>i</sub>x + b<sub>i</sub> is the decision boundary for the i<sup>th</sup> class.


## SVM_TEST_MULTICLASS

```python
y = svm_test_multiclass(W, B, x)
```

This function will take the **Y** decision boundaries as input and a test point **x**, and will return the predicted class, **y**. There are two special cases the function may encounter. The function may find that a test point is in the "rest" for all of our one-vs-rest classifiers, and so it belongs to no class. The function will just return -1 (null) when this happens. We may also find that a test point belongs to two classes. In this case the function will choose the class for which the test point is the farthest from the decision boundary.



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
[test_script]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/SupportVectorMachie/test_script.py
[helpers]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/SupportVectorMachine/helpers.py