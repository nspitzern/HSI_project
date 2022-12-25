"""
Ref:
    [1]	W. Sun, L. Zhang, B. Du, W. Li, and Y. Mark Lai, "Band Selection Using Improved Sparse Subspace Clustering
    for Hyperspectral Imagery Classification," IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing, vol. 8, pp. 2784-2797, 2015.
Formula:
    arg min ||X - XW||_F + lambda||W||_F subject to diag(Z) = 0
Solution:
    Wˆ = −(X^T X + lambda*I)^−1 (diag((X^T X + lambda*I)−1))^−1
"""
from time import perf_counter

import numpy as np
from sklearn.cluster import SpectralClustering


class ISSC_HSI(object):
    """
    :argument:
        Implementation of L2 norm based sparse self-expressive clustering model
        with affinity measurement basing on angular similarity
    """
    def __init__(self, n_band=10, coef_=1):
        self.n_band = n_band
        self.coef_ = coef_

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        """
        :param X: shape [n_row*n_clm, n_band]
        :return: selected band subset
        """
        assert len(X.shape) == 2, "Provide data in the shape: (n_rows*n_cols, n_bands)"
        assert X.shape[1] <= self.n_band, "Number of desired bands can't be larger than the number of bands in the image"

        I = np.eye(X.shape[1])
        coefficient_mat = -1 * np.dot(np.linalg.inv(np.dot(X.transpose(), X) + self.coef_ * I),
                                      np.linalg.inv(np.diag(np.diag(np.dot(X.transpose(), X) + self.coef_ * I))))
        temp = np.linalg.norm(coefficient_mat, axis=0).reshape(1, -1)
        affinity = (np.dot(coefficient_mat.transpose(), coefficient_mat) /
                    np.dot(temp.transpose(), temp))**2

        sc = SpectralClustering(n_clusters=self.n_band, affinity='precomputed')
        sc.fit(affinity)
        selected_band = self.__get_band(sc.labels_, X)
        return selected_band

    def __get_n_clusters(self, W):
        eig_vals, eig_vecs = np.linalg.eig(W)
        ones = np.ones_like(eig_vals)

        DC = np.dot(eig_vals, np.log(np.dot(eig_vecs, ones) ** 2))

        return DC

    def __get_band(self, cluster_result, X):
        """
        select band according to the center of each cluster
        :param cluster_result:
        :param X:
        :return:
        """
        selected_band = []
        n_cluster = np.unique(cluster_result).__len__()
        # img_ = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        for c in np.unique(cluster_result):
            idx = np.nonzero(cluster_result == c)
            center = np.mean(X[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(X[:, idx[0]] - center, axis=0)
            band_ = X[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        bands = np.asarray(selected_band).transpose()
        # bands = bands.reshape(n_cluster, n_row, n_column)
        # bands = np.transpose(bands, axes=(1, 2, 0))
        return bands


if __name__ == '__main__':
    start = perf_counter()
    # a = np.random.randn(50,13)
    # a = np.ones((50,13)) * 0.5
    a = np.random.randint(0, 255, (700*670, 128))
    # print(a)
    w = ISSC_HSI(n_band=10)
    w.fit(a)
    print(w.predict(a))
    end = perf_counter()
    print(f'ETA: {(end - start) / 60} minutes')