

class BaseAlgorithm:
    def __init__(self, n_bands):
        self.n_bands = n_bands

    def fit(self, X):
        pass

    def check_input(self, X):
        self._check_input_dims(X)
        self._check_num_bands(X)

    def _check_input_dims(self, X):
        if len(X.shape) != 3:
            raise ValueError(f'Image should be of dim 3, got dim {len(X.shape)}')

    def _check_num_bands(self, X):
        X_n_bands = X.shape[-1]
        if X_n_bands < self.n_bands:
            raise ValueError(f'Number of desired bands ({self.n_bands}) should be <= number of given bands ({X_n_bands})')

    def _flat_input(self, X):
        num_bands = X.shape[-1]
        return X.reshape((-1, num_bands))