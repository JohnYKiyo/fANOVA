# fANOVA
A [functional ANOVA](./1710.04725.pdf) is a method of quantifying the sensitivity of input variables to output variables in a system model.

This library supports Python3.7+.

Installation requires [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) and [sklearn](https://scikit-learn.org/stable/). These dependencies are written in `requirements.txt`.

# Installation
Install the library by git-cloning this repository.   
```sh
python setup.py install
```   

Alternatively, you can install the library without git-cloning the repository by entering the following command.   
```sh
pip install git+https://github.com/bird-initiative/fANOVA.git
```

# Example
The following is a simple analysis of variance.
```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.datasets

from functionalanova.fanova import FunctionalANOVA

# dataset
X = pd.DataFrame(sklearn.datasets.load_iris()['data'],columns=sklearn.datasets.load_iris()['feature_names'])
y = pd.DataFrame(sklearn.datasets.load_iris()['target'],columns=['species'])

# run analysis of variance
anova = FunctionalANOVA(X=X,y=y, degree=1)
print(anova.importances)
```

# Uninstallation
```sh
pip uninstall functionalanova
```
