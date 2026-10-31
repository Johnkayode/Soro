import numpy as np
from .base import BaseSuppressor, EPS


class SpectralSubtractionSuppressor(BaseSuppressor):
    """Classic spectral subtraction with gain reconstruction."""

    def __init__(self, *args, noise_beta: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_beta = float(noise_beta)

    def _compute_gain(self, mag: np.ndarray, mag2: np.ndarray):
        noise_mag = np.sqrt(self._noise_psd + EPS)
        clean_mag = np.maximum(
            mag - self.noise_beta * noise_mag,
            self.noise_floor * noise_mag,
        )
        gain = clean_mag / (mag + EPS)
        clean_psd = clean_mag * clean_mag
        post_snr = mag2 / (self._noise_psd + EPS)
        return gain, clean_psd, post_snr
