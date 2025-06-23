import numpy as np
from collections import Counter
import pytz
import pandas as pd
from datetime import datetime

np.random.seed(42)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))
        best_feat, best_thresh = self._best_criteria(X, y, n_features)
        if best_feat is None: return Node(value=self._most_common_label(y))
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)
    def _best_criteria(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in range(n_features):
            for threshold in np.unique(X[:, feat_idx]):
                gain = self._information_gain(y, X[:, feat_idx], threshold)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, threshold
        return split_idx, split_thresh
    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        child_entropy = (len(left_idxs) / len(y)) * self._entropy(y[left_idxs]) + (len(right_idxs) / len(y)) * self._entropy(y[right_idxs])
        return parent_entropy - child_entropy
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    def _entropy(self, y):
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForestClassifier:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100):
        self.n_trees, self.min_samples_split, self.max_depth = n_trees, min_samples_split, max_depth
        self.trees = []
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def _bootstrap_samples(self, X, y):
        idxs = np.random.choice(len(X), len(X), replace=True)
        return X[idxs], y[idxs]
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]
    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees]).T
        return np.array([self._most_common_label(p) for p in preds])

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            return Node(value=np.mean(y))
        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            return Node(value=np.mean(y))
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)
    def _best_split(self, X, y, n_features):
        best_mse = float('inf')
        split_idx, split_thresh = None, None
        for feat_idx in range(n_features):
            for threshold in np.unique(X[:, feat_idx]):
                left_idxs, right_idxs = self._split(X[:, feat_idx], threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0: continue
                mse = (len(left_idxs) * np.var(y[left_idxs]) + len(right_idxs) * np.var(y[right_idxs])) / len(y)
                if mse < best_mse:
                    best_mse, split_idx, split_thresh = mse, feat_idx, threshold
        return split_idx, split_thresh
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForestRegressor:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100):
        self.n_trees, self.min_samples_split, self.max_depth = n_trees, min_samples_split, max_depth
        self.trees = []
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def _bootstrap_samples(self, X, y):
        idxs = np.random.choice(len(X), len(X), replace=True)
        return X[idxs], y[idxs]
    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)

def predict_future(model, current_features, features_loaded):
    predictions = [current_features['Temp']] if 'Temp' in current_features else [current_features['Humidity']]
    feature_data = current_features.copy()
    for i in range(5):
        now = datetime.now(pytz.timezone('Asia/Kathmandu'))
        feature_data['day_of_year'] = now.timetuple().tm_yday
        feature_data['month'] = now.month
        feature_data['day_of_week'] = now.weekday()
        prediction_df = pd.DataFrame([feature_data])[features_loaded]
        next_value = model.predict(prediction_df.values)[0]
        predictions.append(next_value)
        if 'Temp' in current_features:
            feature_data['Temp'] = next_value
        else:
            feature_data['Humidity'] = next_value
    return predictions[1:]