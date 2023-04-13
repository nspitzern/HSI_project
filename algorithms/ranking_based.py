"""
Ref: https://www.researchgate.net/publication/3202233_A_joint_band_prioritization_and_band-decorrelation_approach_to_band_selection_for_hyperspectral_image_classification

"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.special import rel_entr

from algorithms.base_class import BaseAlgorithm
from common_utils.timer import timeit
from common_utils.smoothing_methods import lidstone


class MMCA(BaseAlgorithm):
    def __init__(self, n_bands):
        super(MMCA, self).__init__(n_bands)
        self.n_bands = n_bands

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        super().check_input(X)
        X = super()._flat_input(X)

        num_features, num_bands = X.shape

        clusters = KMeans(self.n_bands, n_init='auto').fit_predict(X)

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
            num_features_c = X_c.shape[0]
            mean_c_hat = (mean_c - m).reshape(num_bands, 1)
            S_B += num_features_c * mean_c_hat.dot(mean_c_hat.T)

        A = np.linalg.inv(S_W).dot(S_B)

        eig_vals, eig_vecs = np.linalg.eig(A)

        idxs = np.argsort(eig_vals)[::-1]

        return _divergence_based_band_selection(X, idxs, eps=1.5)


def _divergence_based_band_selection(X, bands_priorities, eps: float):
    def _get_band_histogram(band):
        return np.histogram(band, bins=256, range=(0, 256))

    def _calculate_bands_dkl(B1, B2):
        return np.sum(rel_entr(B1, B2)) + np.sum(rel_entr(B2, B1))

    highest_priority_band = bands_priorities[0]
    final_group = [highest_priority_band]

    num_bands = X.shape[-1]

    bands_histograms = [_get_band_histogram(X[:, i])[0] for i in range(num_bands)]

    # Smooth histograms to eliminate 0 values
    bands_histograms = [lidstone(hist) for hist in bands_histograms]

    for j in np.arange(1, num_bands):
        current_band_priority = bands_priorities[j]
        should_keep = None

        for band_idx in final_group:
            D = _calculate_bands_dkl(bands_histograms[current_band_priority], bands_histograms[band_idx])

            if D < eps:
                should_keep = False
                break

            if band_idx == final_group[-1]:
                should_keep = True
                break

        # In this case the current band has high information to contribute, and it is added your final group
        if should_keep:
            final_group.append(current_band_priority)

    return final_group


@timeit(num_repeats=5)
def main():
    a = np.random.randint(0, 255, (700, 670, 210))
    w = MMCA(n_bands=10)
    w.fit(a)
    pred = w.predict(a)

    print(len(pred))
    print(pred)


if __name__ == '__main__':
    main()
