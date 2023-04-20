"""
Ref: https://www.researchgate.net/publication/3202233_A_joint_band_prioritization_and_band-decorrelation_approach_to_band_selection_for_hyperspectral_image_classification

"""
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from algorithms.base_class import BaseAlgorithm
from algorithms_utils.timer import timeit
from algorithms_utils.misc import get_band_histogram
from algorithms_utils.information_theory import dkl


class MMCA(BaseAlgorithm):
    def __init__(self, n_classes):
        super(MMCA, self).__init__(n_classes)
        self.n_classes = n_classes

    def fit(self, X):
        self.X = X
        return self

    def predict(
            self,
            X: np.ndarray,
            clusters: List[int] = None,
            eps: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the optimal bands.
        :param X: np array Hyperspectral image.
        :param clusters: (optional) A list of clusters indications for each pixel
        :param eps: Dissimilarity threshold for KL Divergence between each band. This param effects the number of optimal
        bands that are being returned.
        :return: Optimal bands and their indices.
        """
        super().check_input(X)
        X = super()._flat_input(X)

        idxs = self._rank_bands(X, clusters)

        bands = _divergence_based_band_selection(X, idxs, eps=eps)
        return X[:, bands], bands

    def _rank_bands(
            self,
            X: np.ndarray,
            clusters: np.ndarray = None
    ) -> List[int]:
        """
        Implementation is similar to LDA here: https://www.youtube.com/watch?v=9IDXYHhAfGA&t=762s
        :param X: Bands vectors (size: (H*W,N_BANDS))
        :param clusters: List of classification to clusters for each feature.
                        If None, use KMeans to cluster into classes.
        :return: List of idxs sorted by eigenvalues
        """
        num_features, num_bands = X.shape

        if clusters is not None:
            clusters = KMeans(self.n_classes, n_init='auto').fit_predict(X)

        m = np.mean(X, axis=0)
        S_W = np.zeros([num_bands, num_bands])
        S_B = np.zeros([num_bands, num_bands])

        for c in np.unique(clusters):
            X_c = X[clusters == c]
            mean_c = np.mean(X_c, axis=0)

            # calculate S_W
            X_c_hat = X_c - mean_c
            S_W += X_c_hat.T.dot(X_c_hat)

            # calculate S_B
            num_bands_c = X_c.shape[-1]
            mean_c_hat = (mean_c - m).reshape(num_bands, 1)
            S_B += num_bands_c * mean_c_hat.dot(mean_c_hat.T)

        S_W /= (num_bands ** 2)
        S_B /= num_bands

        A = np.linalg.inv(S_W).dot(S_B)

        eig_vals, eig_vecs = np.linalg.eig(A)

        idxs = np.argsort(eig_vals)[::-1]

        return idxs


def _divergence_based_band_selection(
        X: np.ndarray,
        bands_priorities: List[int],
        eps: float
) -> np.ndarray:
    """
    See section IV. in https://www.researchgate.net/publication/3202233_A_joint_band_prioritization_and_band-decorrelation_approach_to_band_selection_for_hyperspectral_image_classification
    :param X: Bands vectors (size: (H*W,N_BANDS))
    :param bands_priorities: A list of indices of the bands sorted in descending priority order.
    :param eps: Dissimilarity threshold for KL Divergence between each band. This param effects the number of optimal
        bands that are being returned.
    :return: np array of optimal bands indices.
    """
    highest_priority_band = bands_priorities[0]
    final_group = [highest_priority_band]

    num_bands = X.shape[-1]

    bands_histograms = [get_band_histogram(X[:, i], density=True) for i in range(num_bands)]

    for j in np.arange(1, num_bands):
        current_band_priority = bands_priorities[j]
        should_keep = None

        for band_idx in final_group:
            D = dkl(bands_histograms[current_band_priority], bands_histograms[band_idx])

            if D < eps:
                should_keep = False
                break

            if band_idx == final_group[-1]:
                should_keep = True
                break

        # In this case the current band has high information to contribute, and it is added your final group
        if should_keep:
            final_group.append(current_band_priority)

    return np.array(final_group)


@timeit(num_repeats=1)
def main():
    a = np.random.randint(0, 255, (700, 670, 210))
    w = MMCA(n_classes=10)
    w.fit(a)
    pred = w.predict(a)
    print(len(pred))
    print(pred)


if __name__ == '__main__':
    main()
