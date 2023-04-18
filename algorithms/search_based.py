"""
Ref: https://www.researchgate.net/publication/224342471_Similarity-Based_Unsupervised_Band_Selection_for_Hyperspectral_Image_Analysis

"""

from typing import List, Tuple

import numpy as np

from algorithms.base_class import BaseAlgorithm
from algorithms_utils.timer import timeit


class LP(BaseAlgorithm):
    def __init__(self, n_bands):
        super(LP, self).__init__(n_bands)
        self.n_bands = n_bands

    def fit(
            self,
            X: np.ndarray
    ) -> BaseAlgorithm:
        self.X = X
        return self

    def predict(
            self,
            X: np.ndarray
    ) -> List:
        super().check_input(X)
        X = super()._flat_input(X)
        cleaned_X = self._preprocess(X)

        num_features = cleaned_X.shape[0]

        # select initial bands
        bands_group, group_idxs = self._select_initial_bands_group(cleaned_X)
        group_size = len(group_idxs)

        # add bands iteratively
        while group_size < self.n_bands:
            new_band_idx = self._choose_new_band(cleaned_X, bands_group, num_features)

            bands_group = np.hstack([bands_group, cleaned_X[:, [new_band_idx]]])

            group_size += 1
            group_idxs.append(new_band_idx)

        return group_idxs

    def _select_initial_bands_group(self, X) -> Tuple[np.array, List]:
        def _get_X_proj(X, idx):
            Q, _ = np.linalg.qr(X[:, idx].reshape((-1, 1)))
            X_proj = Q.T.dot(np.delete(X, idx, axis=1))
            return X_proj

        X_num_bands = X.shape[-1]

        first_band = np.random.choice(X_num_bands)

        found = False
        previous_idx = first_band
        # Get the projection of all other vectors except current band on the perpendicular plane of current band
        # i.e. current band - A_i, get the projection of all other bands A_j (i!=j) on {A_i}_perp
        X_proj = _get_X_proj(X, previous_idx)

        current_idx = np.argmax(X_proj)

        i = 0
        while not found:
            i += 1
            if i > X_num_bands ** 2:
                raise StopIteration('Error, exceeded number of iterations')

            # get the projection of all other vectors except current band on the perpendicular plane of current band
            # i.e. current band - A_i, get the projection of all other bands A_j (i!=j) on {A_i}_perp
            X_proj = _get_X_proj(X, current_idx)

            next_idx = np.argmax(X_proj)

            # Adjust indices since we removed one column
            if next_idx >= current_idx:
                next_idx += 1

            if next_idx == previous_idx:
                found = True
            else:
                previous_idx = current_idx
                current_idx = next_idx

        first_band_idx = previous_idx
        second_band_idx = current_idx
        return X[:, [first_band_idx, second_band_idx]], [first_band_idx, second_band_idx]

    def _choose_new_band(
            self,
            B: np.ndarray,
            bands_group: np.ndarray,
            num_features: int
    ) -> int:
        #       |  |  |   |        |
        # X =   |  1  B1  B2  ...  |
        #       |  |  |   |        |
        X = np.hstack([np.ones([num_features, 1]), bands_group])

        a = np.linalg.inv(X.T.dot(X)).dot(X.T)
        a = a.dot(B)

        B_tag = np.hstack([np.ones([num_features, 1]), bands_group]).dot(a)

        e = np.linalg.norm(B - B_tag, axis=0)

        best_band = np.argmax(e)

        return best_band.item()

    def _preprocess(
            self,
            X: np.ndarray,
            drop_thresh: float = 0.8
    ) -> np.ndarray:
        def _calculate_bands_correlation(a):
            return np.corrcoef(a, rowvar=False)

        def _remove_low_correlation_bands(a):
            bands_corr = _calculate_bands_correlation(a)

            bands_idx_to_keep = []

            for i in range(bands_corr.shape[0]):
                left = bands_corr[i, max(i - 1, 0)]
                right = bands_corr[i, min(i + 1, bands_corr.shape[1] - 1)]
                if np.any(np.array([left, right]) < drop_thresh):
                    continue
                else:
                    bands_idx_to_keep.append(i)

            return a[:, bands_idx_to_keep]

        def _whiten_data(X):
            X_cov = np.cov(X - X.mean(axis=0), rowvar=False, bias=True)
            eig_vals, eig_vecs = np.linalg.eig(X_cov)
            D = np.diag(eig_vals)
            W = np.diag(np.diag(D) ** (-0.5))
            X_white = ((W @ eig_vecs.T) @ X.T).T
            return X_white

        # Step A - Remove Bad Bands with low correlation
        X_good_bands = _remove_low_correlation_bands(X)
        # Step B - White data to remove noise
        X_cleaned = _whiten_data(X_good_bands)

        return X_cleaned


@timeit(num_repeats=1)
def main():
    a = np.ones((700, 670, 210))
    # a = np.random.randint(0, 255, (700, 670, 210))
    w = LP(n_bands=10)
    w.fit(a)
    print(w.predict(a))


if __name__ == '__main__':
    main()
