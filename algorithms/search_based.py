from time import perf_counter
from typing import List, Tuple

import numpy as np
# from pysptools.eea.eea import FIPPI, PPI
# from pysptools.eea import NFINDR


class LP:
    def __init__(self, n_bands):
        self.n_bands = n_bands

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X) -> List:
        num_features = X.shape[0]

        # preprocessing?

        # select initial bands
        bands_group, group_idxs = self._select_initial_bands_group(X)
        group_size = len(group_idxs)

        # add bands iteratively
        while group_size < self.n_bands:
            new_band_idx = self._choose_new_band(X, bands_group, num_features)

            bands_group = np.hstack([bands_group, X[:, [new_band_idx]]])

            group_size += 1
            group_idxs.append(new_band_idx)

        return group_idxs


    def _select_initial_bands_group(self, X) -> Tuple[np.array, List]:
        X_num_bands = X.shape[1]

        first_band = np.random.choice(X_num_bands)
        second_band = np.random.choice(X_num_bands)

        return X[:, [first_band, second_band]], [first_band, second_band]

    def _choose_new_band(self, B, bands_group, num_features) -> int:
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


if __name__ == '__main__':
    start = perf_counter()
    a = np.random.randint(0, 255, (700*670, 210))
    w = LP(n_bands=10)
    w.fit(a)
    print(w.predict(a))
    end = perf_counter()
    print(f'ETA: {(end - start) / 60} minutes')