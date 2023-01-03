from time import perf_counter
from typing import List, Tuple

import numpy as np
from pysptools.eea import FIPPI, NFINDR

from base_class import BaseAlgorithm


end_members_funcs = {
    'fippi': FIPPI,
    'nfindr': NFINDR
}


class LP(BaseAlgorithm):
    def __init__(self, n_bands, end_members_ext_func=''):
        super(LP, self).__init__(n_bands)
        self.n_bands = n_bands
        self.end_members_func = end_members_ext_func.lower()

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X) -> List:
        super().check_input(X)

        try:
            # preprocessing
            end_members = end_members_funcs[self.end_members_func]().extract(X, q=10)
        except KeyError:
            raise ValueError("Error, please select available end members function (FIPPI, NFINDR)")

        num_features = end_members.shape[0]

        # select initial bands
        bands_group, group_idxs = self._select_initial_bands_group(end_members)
        group_size = len(group_idxs)

        # add bands iteratively
        while group_size < self.n_bands:
            new_band_idx = self._choose_new_band(end_members, bands_group, num_features)

            bands_group = np.hstack([bands_group, end_members[:, [new_band_idx]]])

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

        i = 1
        while not found:
            i += 1
            print(f'{i=}')
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
        print(f'Starting idxs: {previous_idx, current_idx}')
        return X[:, [first_band_idx, second_band_idx]], [first_band_idx, second_band_idx]

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
    a = np.random.randint(0, 255, (700, 670, 210))
    w = LP(n_bands=10, end_members_ext_func='FIPPI')
    w.fit(a)
    print(w.predict(a))
    end = perf_counter()
    print(f'ETA: {(end - start) / 60} minutes')