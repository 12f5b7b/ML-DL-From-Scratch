import numpy as np
from sklearn.utils.validation import check_X_y

def euclidean_distance(X_train, X_test):
    """
    Compute the Euclidean distance from every training sample to every test sample

    Arguments:
        X_train, np.array  (num_train_samples, num_features)
        X_test,  np.array  (num_test_samples, num_features)

    Returns:
        dists, np.array (num_test_samples, num_train_samples)
    """

    dists = np.sum((X_test[:, np.newaxis, :] - X_train) ** 2, axis=2) ** .5
    return dists


def manhattan_distance(X_train, X_test):
    """
    Compute the Manhattan distance from every training sample to every test sample

    Arguments:
        X_train, np.array  (num_train_samples, num_features)
        X_test,  np.array  (num_test_samples, num_features)

    Returns:
        dists, np.array (num_test_samples, num_train_samples)
    """

    dists = np.sum(np.abs(X_test[:, np.newaxis, :] - X_train), axis=2)
    return dists


def cosine_distance(X_train, X_test):
    """
    Compute the Cosine distance from every training sample to every test sample

    Arguments:
        X_train, np.array  (num_train_samples, num_features)
        X_test,  np.array  (num_test_samples, num_features)

    Returns:
        dists, np.array (num_test_samples, num_train_samples)
    """
    dot_matrix = np.dot(X_test, X_train.T)
    norms = np.sum(X_test[:, np.newaxis] ** 2, axis=2) ** .5 * np.sum(X_train ** 2, axis=1) ** .5
    dists = 1 - dot_matrix / norms
    return dists


distance_metrics_dict = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine': cosine_distance
}

class KNN:
    """
    K-nearest neighbors algorithm
    """

    def __init__(self, n_neighbors=3, metric='euclidean', mod_type='cls'):
        """
        Arguments:
            n_neighbors, int - Number Neighbours
            metric, str - Metric for compute distance (euclidean, manhattan, cosine)
            mod_type, str - Type of model: regression or classification (cls, reg)
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.mod_type = mod_type

    def get_params(self, deep=True):
        return {'n_neighbors': self.n_neighbors, 'metric': self.metric, 'mod_type': self.mod_type}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_train, y):
        self.metric = distance_metrics_dict[self.metric]
        self.X_train = X_train
        self.y = y
        return self

    def predict(self, X):
        """
        Predict mean of k-nearest neighbors for regression and mode for classification
        Arguments:
            X, np.array (num_test_samples, num_features)

        Returns:
            y_test, np.array (num_test_samples)
        """
        dists = self.metric(self.X_train, X)
        sort_dists = np.apply_along_axis(np.argsort, axis=1, arr=dists)
        nearest_k_ind = sort_dists[:, :self.n_neighbors]
        y_test = np.apply_along_axis(lambda x: np.take(self.y, x), axis=1, arr=nearest_k_ind)
        if self.mod_type == 'cls':
            y_test = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_test)
        elif self.mod_type == 'reg':
            y_test = np.mean(y_test, axis=1)

        return y_test