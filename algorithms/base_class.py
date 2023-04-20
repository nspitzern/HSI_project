from typing import Tuple

import numpy as np


class BaseAlgorithm:
    def __init__(self, n_bands: int):
        self.n_bands = n_bands

    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def check_input(
            self,
            X: np.ndarray
    ):
        """
        Validate the input image in terms of dimensions and number of bands.
        :param X: np array.
        """
        self._check_input_dims(X)
        self._check_num_bands(X)

    def _check_input_dims(
            self,
            X: np.ndarray
    ):
        """
        Images should be of dimension 2 (H*W, BANDS) or 3 (H,W,BANDS).
        :param X: np array hyperspectral image
        """
        if len(X.shape) != 3 and len(X.shape) != 2:
            raise ValueError(f'Image should be of dim 3 or 2, got dim {len(X.shape)}')

    def _check_num_bands(
            self,
            X: np.ndarray
    ):
        """
        Number of requested bands should be less or equal to the number of bands in the given image.
        :param X: np array hyperspectral image
        """
        X_n_bands = X.shape[-1]
        if X_n_bands < self.n_bands:
            raise ValueError(f'Number of desired bands ({self.n_bands}) should be <= number of given bands ({X_n_bands})')

    def _flat_input(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Gets a hyperspectral image of dims (H,W,BANDS) and returns it in dims (H*W, BANDS)
        :param X: np array hyperspectral image
        :return: np array flattened hyperspectral image
        """
        num_bands = X.shape[-1]
        return X.reshape((-1, num_bands))