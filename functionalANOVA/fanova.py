from typing import (
    Tuple,
    List,
    Union,
    Dict,
    Optional,
)

import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestRegressor

from .forest_tree import ForestTree

from copy import deepcopy


class FunctionalANOVA(object):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        search_spaces: Optional[np.ndarray] = None,
        degree: int = 1,
        n_tree: int = 32,
        max_depth: int = 64,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        seed: int = 0,
    ) -> None:

        self._X = deepcopy(X)
        self._y = deepcopy(y)
        self.n_features = X.shape[1]
        self._search_spaces = self._get_search_spaces(search_spaces)
        self._columns = list(self._X.columns.values.astype(str))
        self._f = _Fanova(
            n_trees=n_tree,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            seed=seed)
        assert self.n_features >= degree, f'degree must be less than or equal to feature dimension. but degree:{degree}, n_features:{self.n_features}'
        self._combination_of_features = self._listup_combination_of_features(degree=degree, n_features=self.n_features)
        self._compute_importance()

    def _get_search_spaces(
        self,
        search_spaces: Optional[np.ndarray]
    ) -> np.ndarray:

        spaces = deepcopy(search_spaces)
        if spaces is None:
            spaces = np.column_stack([self._X.min().values, self._X.max().values])
        assert self.n_features == spaces.shape[0], 'X must have same dimension of search_spaces'
        assert spaces.shape[1] == 2, 'search_spaces requires [[MIN, MAX], ..., [MIN, MAX]] ndarray'

        return spaces

    def _listup_combination_of_features(self, degree: int, n_features: int) -> List[Tuple]:
        features_indexes = list(range(n_features))
        combinations = []
        for i in range(1, 1 + degree):
            combinations += list(itertools.combinations(features_indexes, i))
        return combinations

    def _compute_importance(self) -> None:
        self._f.fit(X=self._X.values,
                    y=self._y.values.ravel(),
                    search_spaces=self._search_spaces,
                    column_to_encoded_columns=self._combination_of_features)

        importances_value = [self._f.get_importance((i,))[0] for i in range(len(self._combination_of_features))]
        importances_error = [self._f.get_importance((i,))[1] for i in range(len(self._combination_of_features))]
        name_list = []
        for comb_name in self._combination_of_features:
            name = ''
            for i in comb_name:
                name += list(self._columns)[i] + ' '
            name_list.append(name[:-1])

        self._importances = pd.DataFrame(
            {
                'importance_value': importances_value,
                'importance_error': importances_error,
                'marginal_feature_name': name_list
            })

    @property
    def importances(self) -> pd.DataFrame:
        return self._importances

    def get_importances(self, features: Tuple[int, ...]) -> Tuple[float, float]:
        return self._f.get_importance(features)


class _Fanova(object):
    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        min_samples_split: Union[int, float],
        min_samples_leaf: Union[int, float],
        seed: Optional[int],
    ) -> None:

        self._forest = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        self._trees: Optional[List[ForestTree]] = None
        self._variances: Optional[Dict[Tuple[int, ...], np.ndarray]] = None
        self._column_to_encoded_columns: Optional[List[np.ndarray]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        search_spaces: np.ndarray,
        column_to_encoded_columns: List[Tuple],
    ) -> None:
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == search_spaces.shape[0]
        assert search_spaces.shape[1] == 2

        self._forest.fit(X, y)

        self._trees = [ForestTree(e.tree_, search_spaces) for e in self._forest.estimators_]
        self._column_to_encoded_columns = column_to_encoded_columns
        self._variances = {}

        if all(tree.variance == 0 for tree in self._trees):
            # If all trees have 0 variance, we cannot assess any importances.
            # This could occur if for instance `X.shape[0] == 1`.
            raise RuntimeError("Encountered zero total variance in all trees.")

    def get_importance(self, features: Tuple[int, ...]) -> Tuple[float, float]:
        # Assert that `fit` has been called.
        assert self._trees is not None
        assert self._variances is not None

        self._compute_variances(features)

        fractions: Union[List[float], np.ndarray] = []

        for tree_index, tree in enumerate(self._trees):
            tree_variance = tree.variance
            if tree_variance > 0.0:
                fraction = self._variances[features][tree_index] / tree_variance
                fractions = np.append(fractions, fraction)

        fractions = np.asarray(fractions)

        return float(fractions.mean()), float(fractions.std())

    def _compute_variances(self, features: Tuple[int, ...]) -> None:
        assert self._trees is not None
        assert self._variances is not None
        assert self._column_to_encoded_columns is not None

        if features in self._variances:
            return

        for k in range(1, len(features)):
            for sub_features in itertools.combinations(features, k):
                if sub_features not in self._variances:
                    self._compute_variances(sub_features)

        raw_features = np.concatenate([self._column_to_encoded_columns[f] for f in features])

        variances = np.empty(len(self._trees), dtype=np.float64)

        for tree_index, tree in enumerate(self._trees):
            marginal_variance = tree.get_marginal_variance(raw_features)

            # See `fANOVA.__compute_marginals` in https://github.com/automl/fanova/blob/master/fanova/fanova.py.
            for k in range(1, len(features)):
                for sub_features in itertools.combinations(features, k):
                    marginal_variance -= self._variances[sub_features][tree_index]

            variances[tree_index] = np.clip(marginal_variance, 0.0, np.inf)

        self._variances[features] = variances
