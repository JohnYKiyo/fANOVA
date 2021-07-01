from typing import (
    TYPE_CHECKING,
    Tuple,
    List,
    Union,
    Set
)

import numpy as np
import itertools

if TYPE_CHECKING:
    from sklearn.tree._tree import Tree


class ForestTree(object):
    def __init__(self, tree: 'Tree', search_spaces: np.ndarray) -> None:
        """AI is creating summary for __init__

        Args:
            tree (Tree): [description]
            search_spaces (ndarray): [description]
        """
        assert tree.n_features == search_spaces.shape[0], 'tree must have same dimension of search_spaces'
        assert search_spaces.shape[1] == 2, 'search_spaces requires [[MIN, MAX], ..., [MIN, MAX]] ndarray'

        self._forest_tree = tree
        self._search_spaces = search_spaces

        node_values_weights = self._calc_node_values_weights()
        split_midpoints, split_sizes = self._precompute_split_midpoints_and_sizes()
        subtree_active_features = self._precompute_subtree_active_features()

        self._node_values_weights = node_values_weights
        self._split_midpoints = split_midpoints
        self._split_sizes = split_sizes
        self._subtree_active_features = subtree_active_features
        self._variance = None

    def _calc_node_values_weights(self) -> np.ndarray:
        n_nodes = self._n_nodes

        node_values_weights = np.empty((n_nodes, 2), dtype=np.float64)

        # Compute marginals for leaf nodes.
        subspaces = np.array([None] * n_nodes)
        subspaces[0] = self._search_spaces  # initial subspaces
        for node_index in range(n_nodes):
            subspace = subspaces[node_index]

            if self._is_node_leaf(node_index):
                value = self._get_node_value(node_index)
                weight = self._get_cardinality(subspace)
                node_values_weights[node_index] = [value, weight]
            else:
                for child_node_index, child_subspace in zip(
                    self._get_node_children(node_index),
                    self._get_node_children_subspaces(node_index, subspace),
                ):
                    assert subspaces[child_node_index] is None
                    subspaces[child_node_index] = child_subspace

        # Compute marginals for internal nodes.
        for node_index in reversed(range(n_nodes)):
            if not self._is_node_leaf(node_index):
                child_values = []
                child_weights = []
                for child_node_index in self._get_node_children(node_index):
                    child_values.append(node_values_weights[child_node_index, 0])
                    child_weights.append(node_values_weights[child_node_index, 1])
                value = np.average(child_values, weights=child_weights)
                weight = float(np.sum(child_weights))
                node_values_weights[node_index] = [value, weight]

        return node_values_weights

    @property
    def variance(self) -> float:
        if self._variance is None:
            leaf_node_indices = np.array(
                [
                    node_index
                    for node_index in range(self._n_nodes)
                    if self._is_node_leaf(node_index)
                ]
            )
            node_values_weights = self._node_values_weights[leaf_node_indices]
            values = node_values_weights[:, 0]
            weights = node_values_weights[:, 1]
            average_values = np.average(values, weights=weights)
            variance = np.average((values - average_values) ** 2, weights=weights)
            self._variance = variance

        assert self._variance is not None
        return self._variance

    def get_marginal_variance(self, features: np.ndarray) -> float:
        assert features.size > 0

        # For each midpoint along the given dimensions, traverse this tree to compute the
        # marginal predictions.
        midpoints = [self._split_midpoints[f] for f in features]
        sizes = [self._split_sizes[f] for f in features]

        product_midpoints = itertools.product(*midpoints)
        product_sizes = itertools.product(*sizes)

        sample = np.full(self._n_features, fill_value=np.nan, dtype=np.float64)

        values: Union[List[float], np.ndarray] = []
        weights: Union[List[float], np.ndarray] = []

        for midpoints, sizes in zip(product_midpoints, product_sizes):
            sample[features] = np.array(midpoints)

            value, weight = self._get_marginalized_node_values_weights(sample)
            weight *= float(np.prod(sizes))

            values = np.append(values, value)
            weights = np.append(weights, weight)

        weights = np.asarray(weights)
        values = np.asarray(values)
        average_values = np.average(values, weights=weights)
        variance = np.average((values - average_values) ** 2, weights=weights)

        assert variance >= 0.0
        return variance

    def _get_marginalized_node_values_weights(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        assert feature_vector.size == self._n_features

        marginalized_features = np.isnan(feature_vector)
        active_features = ~marginalized_features

        search_spaces = self._search_spaces.copy()
        search_spaces[marginalized_features] = [0.0, 1.0]

        active_nodes = [0]
        active_search_spaces = [search_spaces]

        node_indices = []
        active_features_cardinalities = []

        while len(active_nodes) > 0:
            node_index = active_nodes.pop()
            search_spaces = active_search_spaces.pop()

            feature = self._get_node_split_feature(node_index)
            if feature >= 0:
                response = feature_vector[feature]
                if not np.isnan(response):
                    if response <= self._get_node_split_threshold(node_index):
                        next_node_index = self._get_node_left_child(node_index)
                        next_subspace = self._get_node_left_child_subspaces(
                            node_index, search_spaces
                        )
                    else:
                        next_node_index = self._get_node_right_child(node_index)
                        next_subspace = self._get_node_right_child_subspaces(
                            node_index, search_spaces
                        )

                    active_nodes.append(next_node_index)
                    active_search_spaces.append(next_subspace)
                    continue

                if (active_features & self._subtree_active_features[node_index]).any():
                    for child_node_index in self._get_node_children(node_index):
                        active_nodes.append(child_node_index)
                        active_search_spaces.append(search_spaces)
                    continue

            node_indices.append(node_index)
            active_features_cardinalities.append(self._get_cardinality(search_spaces))

        node_values_weights = self._node_values_weights[node_indices]
        values = node_values_weights[:, 0]
        weights = node_values_weights[:, 1]
        weights = weights / active_features_cardinalities

        value = np.average(values, weights=weights)
        weight = weights.sum()

        return value, weight

    def _precompute_split_midpoints_and_sizes(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        midpoints = []
        sizes = []

        search_spaces = self._search_spaces
        for feature, feature_split_values in enumerate(self._compute_features_split_values()):
            feature_split_values = np.concatenate(
                (
                    np.atleast_1d(search_spaces[feature, 0]),
                    feature_split_values,
                    np.atleast_1d(search_spaces[feature, 1]),
                )
            )
            midpoint = 0.5 * (feature_split_values[1:] + feature_split_values[:-1])
            size = feature_split_values[1:] - feature_split_values[:-1]

            midpoints.append(midpoint)
            sizes.append(size)

        return midpoints, sizes

    def _compute_features_split_values(self) -> List[np.ndarray]:
        all_split_values: List[Set[float]] = [set() for _ in range(self._n_features)]

        for node_index in range(self._n_nodes):
            feature = self._get_node_split_feature(node_index)
            if feature >= 0:
                threshold = self._get_node_split_threshold(node_index)
                all_split_values[feature].add(threshold)

        sorted_all_split_values: List[np.ndarray] = []

        for split_values in all_split_values:
            split_values_array = np.array(list(split_values), dtype=np.float64)
            split_values_array.sort()
            sorted_all_split_values.append(split_values_array)

        return sorted_all_split_values

    def _precompute_subtree_active_features(self) -> np.ndarray:
        subtree_active_features = np.full((self._n_nodes, self._n_features), fill_value=False)

        for node_index in reversed(range(self._n_nodes)):
            feature = self._get_node_split_feature(node_index)
            if feature >= 0:
                subtree_active_features[node_index, feature] = True
                for child_node_index in self._get_node_children(node_index):
                    subtree_active_features[node_index] |= subtree_active_features[child_node_index]

        return subtree_active_features

    @property
    def _n_nodes(self) -> int:
        return self._forest_tree.node_count

    @property
    def _n_features(self) -> int:
        return len(self._search_spaces)

    def _is_node_leaf(self, node_index: int) -> bool:
        return self._forest_tree.feature[node_index] < 0

    def _get_node_left_child(self, node_index: int) -> int:
        return self._forest_tree.children_left[node_index]

    def _get_node_right_child(self, node_index: int) -> int:
        return self._forest_tree.children_right[node_index]

    def _get_node_children(self, node_index: int) -> Tuple[int, int]:
        return self._get_node_left_child(node_index), self._get_node_right_child(node_index)

    def _get_node_value(self, node_index: int) -> float:
        return self._forest_tree.value[node_index]

    def _get_node_split_threshold(self, node_index: int) -> float:
        return self._forest_tree.threshold[node_index]

    def _get_node_split_feature(self, node_index: int) -> int:
        return self._forest_tree.feature[node_index]

    def _get_node_left_child_subspaces(
        self,
        node_index: int,
        search_spaces: np.ndarray
    ) -> np.ndarray:

        return self._get_subspaces(
            search_spaces,
            search_spaces_column=1,
            feature=self._get_node_split_feature(node_index),
            threshold=self._get_node_split_threshold(node_index),
        )

    def _get_node_right_child_subspaces(
        self,
        node_index: int,
        search_spaces: np.ndarray
    ) -> np.ndarray:
        return self._get_subspaces(
            search_spaces,
            search_spaces_column=0,
            feature=self._get_node_split_feature(node_index),
            threshold=self._get_node_split_threshold(node_index),
        )

    def _get_node_children_subspaces(
        self,
        node_index: int,
        search_spaces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self._get_node_left_child_subspaces(node_index, search_spaces),
            self._get_node_right_child_subspaces(node_index, search_spaces),
        )

    def _get_subspaces(
        self,
        search_spaces: np.ndarray,
        search_spaces_column: int,
        feature: int,
        threshold: float,
    ) -> np.ndarray:
        search_spaces_subspace = np.copy(search_spaces)
        search_spaces_subspace[feature, search_spaces_column] = threshold
        return search_spaces_subspace

    def _get_cardinality(self, search_spaces: np.ndarray) -> float:
        return np.prod(search_spaces[:, 1] - search_spaces[:, 0])  # \Pi_i^d (max_i - min_i)
