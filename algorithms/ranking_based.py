from time import perf_counter
from typing import List

import numpy as np
from sklearn.cluster import KMeans


class MMCA:
    def __init__(self, n_bands):
        self.n_bands = n_bands

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X) -> List[int, ...]:
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

        return idxs[:self.n_bands]


if __name__ == '__main__':
    start = perf_counter()
    a = np.random.randint(0, 255, (700*670, 210))
    w = MMCA(n_bands=20)
    w.fit(a)
    print(w.predict(a))
    end = perf_counter()
    print(f'ETA: {(end - start) / 60} minutes')