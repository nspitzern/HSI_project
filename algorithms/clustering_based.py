"""
Ref: https://www.researchgate.net/publication/3205663_Clustering-Based_Hyperspectral_Band_Selection_Using_Information_Measures

"""

import itertools
from typing import Tuple

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from scipy.special import entr

from algorithms.base_class import BaseAlgorithm
from algorithms_utils.timer import timeit
from algorithms_utils.information_theory import dkl
from algorithms_utils.misc import get_band_histogram


class WALU(BaseAlgorithm):
    def __init__(self, n_bands, similarity_func):
        super(WALU, self).__init__(n_bands)
        self.similarity_func = similarity_func
        self.hc = AgglomerativeClustering(n_clusters=n_bands)

    def _select_bands(
            self,
            D: np.ndarray,
            clusters: np.ndarray
    ) -> np.ndarray:
        """
        Returns the representative bands of each cluster
        :param D: Bands similarity matrix.
        :param clusters: Cluster belonging of each band.
        :return: np array of bands indices.
        """
        clusters_representatives = []

        for c in np.unique(clusters):
            cluster_c = np.argwhere(clusters == c).flatten()
            R = len(cluster_c)
            W = np.zeros(R)

            for i_idx, band_i in enumerate(cluster_c):
                sum_cluster = 0
                for band_j in cluster_c:
                    if band_i != band_j:
                        sum_cluster += 1 / (D[band_i, band_j] ** 2 + 1e-10)

                W[i_idx] = sum_cluster / R

            rep_idx = np.argmax(W)
            clusters_representatives.append(cluster_c[rep_idx])
        return np.array(clusters_representatives)

    def fit(self, X: np.ndarray):
        self.X = X
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        super().check_input(X)
        X = super()._flat_input(X)

        D = self.similarity_func(X)
        clusters = self.hc.fit_predict(D)

        bands = self._select_bands(D, clusters)
        return X[:, bands], bands


class WALUDI(WALU):
    def __init__(self, n_bands):
        """
        Implementation of WALUDI algorithm for optimal bands selection
        :param n_bands: number of desired bands
        """
        super(WALUDI, self).__init__(n_bands, self._calculate_kl_divergence_score)

    def _calculate_kl_divergence_score(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the KL Divergence between each 2 bands
        :param X: Image as np array (H*W, BANDS)
        :return: np array of size (BANDS, BANDS) where each cell i,j is the KL Divergence between band i and band j.
        """
        num_bands = X.shape[-1]
        bands_histograms = [get_band_histogram(X[:, i], density=True) for i in range(num_bands)]

        # Get the D_kl for each 2 bands in the image
        kls = [dkl(bands_histograms[idxs[0]], bands_histograms[1])
               for idxs in itertools.product(range(num_bands), range(num_bands))]

        kls = np.array(kls).reshape(num_bands, num_bands)

        return kls


class WALUMI(WALU):
    def __init__(self, n_bands):
        """
        Implementation of WALUMI algorithm for optimal bands selection
        :param n_bands: Number of desired clusters
        """
        super(WALUMI, self).__init__(n_bands, self._calculate_normalized_mutual_info_distance)

    def _calculate_normalized_mutual_info_distance(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the Normalized Mutual Information Distance between each 2 bands
        :param X: Image as np array (H*W, BANDS)
        :return: np array of size (BANDS, BANDS) where each cell i,j is the norm-MI distance between band i and band j.
        """
        n_bands = X.shape[-1]
        X_hat = np.empty((n_bands, n_bands))

        for i in range(n_bands):
            for j in range(i, n_bands):
                band1 = get_band_histogram(X[:, i], density=True)
                band2 = get_band_histogram(X[:, j], density=True)

                H1 = np.sum(entr(band1))  # Entropy X_i
                H2 = np.sum(entr(band2))  # Entropy X_j
                c_xy = np.histogram2d(band1, band2)[0]
                H_1_2 = np.sum(entr(c_xy))
                I = H1 + H2 - H_1_2  # Mutual Information X_i, X_j
                # I = mutual_info_score(None, None, contingency=c_xy)  # Mutual Information X_i, X_j

                NI = (2 * I) / (H1 + H2)  # Normalized Mutual Information
                DNI = (1 - np.sqrt(NI)) ** 2  # NI distance

                X_hat[i, j] = DNI
                X_hat[j, i] = DNI

        return X_hat


@timeit()
def _main():
    a = np.random.randint(0, 255, (700, 670, 128))
    w = WALUMI(n_bands=10)
    w.fit(a)
    print(w.predict(a))


if __name__ == '__main__':
    _main()

