[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<h1 align="center">K-Means Clustering</h1>

## About this Project
Developed Functions:
1. [K_Means(X, K, mu)](#k_means)
2. [K_Means_better(X, K)](#k_means_better)

There is also a test script called [test_script.py](#test_script) which can be used to check the working of the functions

## K_MEANS

```python
C = K_Means(X, K, mu)
```

This function takes the feature vectors **X** and a **K** value as input and returns a numpy array of cluster centers **C**. The function will be able to handle any dimension of feature vectors and any **K > 0**. **mu** is an array of initial cluster centers, with either K or 0 rows. If mu is empty, then the function initializes the cluster centers ran-
domly. Otherwise, it starts with the given cluster centers.


## K_MEANS_BETTER

```python
C = K_Means_better(X, K)
```

This function takes the feature vectors **X** and a **K** value as input and returns a numpy array of cluster centers **C**. The function can handle any dimension of feature vectors and any **K > 0**. The function will run the above-implemented **K_Means(X,K,[])** function many times until the same set of cluster centers are returned a majority of the time. At this point, we will know that those cluster centers are likely the best ones. **K_Means_better(X,K)** will return those cluster centers.


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
[test_script]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/KMeansClustering/test_script.py