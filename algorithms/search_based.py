import numpy as np
from pysptools.eea.eea import FIPPI, PPI
from pysptools.eea import NFINDR


class LP:
    def __init__(self, n_bands):
        self.n_bands = n_bands

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        endmembers = FIPPI(X)

    def _band_selection(self):
        pass
