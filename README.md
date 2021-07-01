# fANOVA
fANOVA implementation.

This library supports Python3.7+.

Installation requires [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) and [sklearn](https://scikit-learn.org/stable/). These dependencies are written in `requirements.txt`.

# Installation
```sh
python setup.py install
```

# Example
The following is a simple analysis of variance.
```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.datasets

from functionalANOVA.fanova import FunctionalANOVA

# dataset
X = pd.DataFrame(sklearn.datasets.load_iris()['data'],columns=sklearn.datasets.load_iris()['feature_names'])
y = pd.DataFrame(sklearn.datasets.load_iris()['target'],columns=['species'])

# run analysis of variance
anova = FunctionalANOVA(X=X,y=y, degree=1)
print(anova.importances)
```

# Uninstallation
```sh
pip uninstall functionalANOVA
```
