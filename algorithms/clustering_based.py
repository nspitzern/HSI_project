import itertools
from time import perf_counter

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from scipy.special import kl_div, entr, softmax

from base_class import BaseAlgorithm
from common_utils.timer import timeit


class WALU(BaseAlgorithm):
    def __init__(self, n_bands):
        super(WALU, self).__init__(n_bands)

    def cluster(self, X):
        pass

    def select_bands(self, D, clusters):
        clusters_representatives = []

        for c in np.unique(clusters):
            cluster_c = np.argwhere(clusters == c)
            R = len(cluster_c)
            W = np.zeros(R)

            for i in range(R):
                sum_cluster = 0
                for idx in cluster_c:
                    if i != idx:
                        sum_cluster += 1 / (D[i, idx[0]] ** 2 + 1e-10)

                W[i] = sum_cluster / R

            rep_idx = np.argmax(W)
            clusters_representatives.append(cluster_c[rep_idx])
        return np.array(clusters_representatives).reshape(-1, 1)


class WALUDI(WALU):
    def __init__(self, n_bands):
        super(WALUDI, self).__init__(n_bands)
        self.hc = AgglomerativeClustering(n_clusters=self.n_bands)

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        # super().check_input(X)
        # X = super()._flat_input(X)

        D = self._calculate_kl_divergence_score(X)
        clusters = self.hc.fit_predict(D)

        return self.select_bands(D, clusters)

    def _calculate_kl_divergence_score(self, X):
        X_hist = softmax(X)
        return kl_div(X_hist, X_hist)


class WALUMI(WALU):
    def __init__(self, n_bands):
        """

        :param n_bands: Number of desired clusters
        """
        super(WALUMI, self).__init__(n_bands)
        self.hc = AgglomerativeClustering(n_clusters=self.n_bands)

    def fit(self, X):
        self.X = X
        return self

        # clusters = {c: [c] for c in range(D.shape[0])}
        #
        # while D.shape[0] > self.n_bands:
        #     # Get 2 most similar clusters
        #     cluster1, cluster2 = np.unravel_index(np.argmin(D), D.shape)
        #
        #     # Update D
        #     D_1_2 = D[cluster1, cluster2]
        #     D_new_cluster = np.zeros(D.shape[0] - 2)
        #     for i in np.arange(D.shape[0]):
        #         if i == cluster1 or i == cluster2:
        #             continue
        #
        #         D_i_1 = D[i, cluster1]
        #         D_i_2 = D[i, cluster2]
        #
        #         # calculate alpha, beta, gamma, delta
        #         n_i = len(clusters[i])
        #         n_1 = len(clusters[cluster1])
        #         n_2 = len(clusters[cluster2])
        #
        #         total_n = n_1 + n_2 + n_i
        #
        #         alpha = (n_1 + n_i) / total_n
        #         beta = (n_2 + n_i) / total_n
        #         gamma = -1 * (n_i) / total_n
        #
        #         D_new = alpha * D[i, cluster1] + beta * D[i, cluster2] + gamma * D_1_2
        #         D_new_cluster[i] = D_new
        #
        #     # Delete old clusters from D
        #     D = np.delete(D, cluster1, axis=0)
        #     D = np.delete(D, cluster2, axis=0)
        #
        #     D = np.delete(D, cluster1, axis=1)
        #     D = np.delete(D, cluster2, axis=1)
        #
        #     # Add new cluster to D
        #     np.resize(D, (D.shape[0] + 1, D.shape[1] + 1))
        #     D[-1, :-1] = D_new_cluster
        #     D[:-1, -1] = D_new_cluster
        #     D[-1, -1] = np.inf

    def predict(self, X):
        super().check_input(X)
        X = super()._flat_input(X)

        D = self._calculate_normalized_mutual_info_distance(X)
        clusters = self.hc.fit_predict(D)

        return self.select_bands(D, clusters)


    def _update_clusters_distance(self, merged_idx1, merged_idx2):
        pass

    def _calculate_normalized_mutual_info_distance(self, X):
        X_hist = softmax(X, axis=0)
        n_bands = X_hist.shape[1]
        X_hat = np.empty((n_bands, n_bands))

        for i, j in itertools.product(range(n_bands), range(n_bands)):
            band1 = X_hist[:, i]
            band2 = X_hist[:, j]

            H1 = np.sum(entr(band1))  # entropy X_i
            H2 = np.sum(entr(band2))  # Entropy X_j
            c_xy = np.histogram2d(band1, band2)[0]
            I = mutual_info_score(None, None, contingency=c_xy)  # Mutual Information X_i, X_j
            # I = mutual_info_score(band1, band2)

            NI = (2 * I) / (H1 + H2)  # Normalized Mutual Information
            DNI = (1 - np.sqrt(NI)) ** 2  # NI distance

            X_hat[i, j] = DNI

        return X_hat

        # X_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=np.arange(X.min(), X.max()))[0], 0, X)
        # X_hist = softmax(X, axis=0)
        #
        # H = entr(X_hist).sum(axis=0)
        #
        # I = self._calculate_mi(X)
        #
        # NI = (2 * I) / (H + H)
        #
        # DNI = (1 - np.sqrt(NI)) ** 2
        # return DNI

    def _calculate_mi(self, X):
        # calculate the probability of X_i conditional on X_j
        row_totals = X.sum(axis=1).astype(float)
        X_cond = (X.T / row_totals).T

        # Calculate the probability of X_i
        col_totals = X.sum(axis=0).astype(float)
        prob_X = col_totals / sum(col_totals)

        # Calculate the joint probability of X_i and X_j
        joint_prob = X_cond / prob_X
        joint_prob[joint_prob==0] = 0.00001
        _pmi = (np.log(joint_prob) * (prob_X * prob_X))
        _pmi[_pmi < 0] = 0

        return _pmi.sum(axis=0)


@timeit()
def main():
    a = np.random.randint(0, 255, (700, 670, 128))
    w = WALUMI(n_bands=3)
    w.fit(a)
    print(w.predict(a))


if __name__ == '__main__':
    main()

