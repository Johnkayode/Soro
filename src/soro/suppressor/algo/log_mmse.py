import numpy as np
from scipy.special import expn
from .base import BaseSuppressor, EPS


class LogMMSESuppressor(BaseSuppressor):
    """Log-MMSE magnitude estimator with decision-directed prior SNR."""

    def __init__(self, *args, dd_alpha: float = 0.98, **kwargs):
        super().__init__(*args, **kwargs)
        self.dd_alpha = float(dd_alpha)

    def _compute_gain(self, mag: np.ndarray, mag2: np.ndarray):
        post_snr = mag2 / (self._noise_psd + EPS)
        prior_snr = self.dd_alpha * (self._prev_clean_psd / (self._noise_psd + EPS))
        prior_snr += (1.0 - self.dd_alpha) * np.maximum(post_snr - 1.0, 0.0)

        prior_snr = np.maximum(prior_snr, 1e-4)
        v = (prior_snr * post_snr) / (1.0 + prior_snr)
        v = np.maximum(v, 1e-4)

        # G_logmmse = (xi/(1+xi)) * exp(0.5 * E1(v))
        g = (prior_snr / (1.0 + prior_snr)) * np.exp(0.5 * expn(1, v))
        g = np.minimum(g, 1.0)

        clean_psd = g * g * mag2
        return g, clean_psd, post_snr
