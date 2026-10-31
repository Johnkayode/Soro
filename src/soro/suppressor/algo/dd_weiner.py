import numpy as np

from .base import BaseSuppressor, EPS


class DDWienerSuppressor(BaseSuppressor):
    """Decision-directed Wiener filtering."""

    def __init__(self, *args, dd_alpha: float = 0.98, **kwargs):
        super().__init__(*args, **kwargs)
        self.dd_alpha = float(dd_alpha)

    def _compute_gain(self, mag: np.ndarray, mag2: np.ndarray):
        post_snr = mag2 / (self._noise_psd + EPS)
        prior_snr = self.dd_alpha * (self._prev_clean_psd / (self._noise_psd + EPS))
        prior_snr += (1.0 - self.dd_alpha) * np.maximum(post_snr - 1.0, 0.0)

        gain = prior_snr / (1.0 + prior_snr)
        clean_psd = gain * gain * mag2
        return gain, clean_psd, post_snr
