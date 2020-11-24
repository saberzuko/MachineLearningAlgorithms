[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<p align="center">
<h1 align="center">Principal Component Analysis</h1>
</p>


## About the Project
Developed functions:
1. [compute_Z](#compute_z)
2. [compute_covariance_matrix](#compute_covariance_matrix)
3. [find_pcs](#find_pcs)
4. [project_data](#project_data)

There is also a test script called [test_script.py][test_script] which can be used to check the working of the functions


## COMPUTE_Z

```python
Z = compute_Z(X, centering=True, scaling=False)
```

This function will take the data matrix **X**, and boolean variables **centering** and **scaling**. **X** has one sample per row. Remember there are no labels in PCA. If **centering** is True, the function will subtract the mean from each feature. If **scaling** is True, the function will divide each feature by its standard deviation. This function will return the **Z** matrix (numpy array), which is the same size as **X**.


## COMPUTE_COVARIANCE_MATRIX

```python
COV = compute_covariance_matrix(Z)
```

This function will take the standardized data matrix Z and return the covariance matrix Z<sup>T</sup>Z=COV (a numpy array).


## FIND_PCS

```python
(L, PCS) = find_pcs(COV)
```

This function will take the covariance matrix **COV** and return the ordered (largest to smallest) principal components **PCS** (a numpy array where each column is an eigenvector) and corresponding eigenvalues **L** (a numpy array).


## PROJECT_DATA

```python
Z_star = project_data(Z, PCS, L, k, var)
```

This function will take the standardized data matrix **Z**, the principal components **PCS**, and corresponding eigenvalues **L**, as well as a **k** integer value and a **var** floating point value. **k** is the number of principal components you wish to maintain when projecting the data into the new space. 0 &le; **k** &ge; **D**. If **k = 0**, then we use the cumulative variance to determine the projection dimension. **var** is the desired cumulative variance explained by the projection. 0 &le; **v** &le; 1. If **v = 0**, then **k** is used instead. Both **k** and **var** should never be 0 but both can be > 0. This function will return **Z_star**, the projected data.


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
[test_script]: https://github.com/saberzuko/MachineLearningAlgorithms/blob/master/PrincipalComponentAnalysis/test_script.py