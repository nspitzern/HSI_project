from typing import Dict
from random import choice

import numpy as np


def get_band_histogram(
        band: np.ndarray,
        density: bool = False
) -> np.ndarray:
    return np.histogram(band, bins=256, range=(0, 256), density=density)[0]


class Cache:
    def __init__(self, max_size):
        self.__max_size = max_size
        self.__buffer: Dict = dict()

    def __setitem__(self, key, val):
        if len(self.__buffer.keys()) >= self.__max_size:
            k = choice(list(self.__buffer.keys()))
            del self.__buffer[k]

        self.__buffer[key] = val

    def __getitem__(self, item) -> float:
        return self.__buffer[item]

    def get(self, item) -> float:
        return self.__buffer.get(item)
