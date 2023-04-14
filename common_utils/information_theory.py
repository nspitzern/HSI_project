from typing import Union

import numpy as np
from numpy import ndarray


def kl(
        p: np.ndarray,
        q: np.ndarray
) -> Union[int, ndarray]:
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.array(p, dtype=np.float)
    q = np.array(q, dtype=np.float)

    if np.all(p == q):
        return 0

    return np.sum(np.where((q != 0) & (p != 0), p * np.log(p / q), 0))


def dkl(
        p: np.ndarray,
        q: np.ndarray
) -> Union[int, ndarray]:
    return kl(p, q) + kl(q, p)