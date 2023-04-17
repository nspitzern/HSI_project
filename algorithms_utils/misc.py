import numpy as np


def get_band_histogram(
        band: np.ndarray,
        density: bool = False
) -> np.ndarray:
    return np.histogram(band, bins=256, range=(0, 256), density=density)[0]