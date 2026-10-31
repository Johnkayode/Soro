import numpy as np
from .base import BaseSuppressor, EPS

class OMLSASuppressor(BaseSuppressor):
    """OM-LSA-style gain with minima-controlled recursive noise tracking."""

    def __init__(
        self,
        *args,
        dd_alpha: float = 0.98,
        omlsa_q: float = 0.3,
        omlsa_min_gain_db: float = -20.0,
        min_track_frames: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dd_alpha = float(dd_alpha)
        self.omlsa_q = float(np.clip(omlsa_q, 0.01, 0.99))
        self.omlsa_min_gain = float(10.0 ** (omlsa_min_gain_db / 20.0))
        self.min_track_frames = max(5, int(min_track_frames))
        self._smooth_psd = self._noise_psd.copy()
        self._min_psd = self._noise_psd.copy()
        self._frame_idx = 0

    def _update_noise_psd(self, mag2: np.ndarray):
        self._frame_idx += 1
        self._smooth_psd = 0.8 * self._smooth_psd + 0.2 * mag2

        if self._frame_idx % self.min_track_frames == 1:
            self._min_psd = self._smooth_psd.copy()
        else:
            self._min_psd = np.minimum(self._min_psd, self._smooth_psd)

        post_snr = mag2 / (self._noise_psd + EPS)
        prior_snr = self.dd_alpha * (self._prev_clean_psd / (self._noise_psd + EPS))
        prior_snr += (1.0 - self.dd_alpha) * np.maximum(post_snr - 1.0, 0.0)

        llr = (post_snr * prior_snr / (1.0 + prior_snr)) - np.log1p(prior_snr)
        odds = (self.omlsa_q / (1.0 - self.omlsa_q)) * np.exp(-np.clip(llr, -30.0, 30.0))
        p_speech = 1.0 / (1.0 + odds)

        alpha = self.ema_alpha + (self.speech_alpha - self.ema_alpha) * p_speech
        self._noise_psd = alpha * self._noise_psd + (1.0 - alpha) * mag2

        # Keep the estimate near local minima to resist speech leakage.
        self._noise_psd = np.maximum(self._noise_psd, 0.9 * self._min_psd)

    def _compute_gain(self, mag: np.ndarray, mag2: np.ndarray):
        post_snr = mag2 / (self._noise_psd + EPS)
        prior_snr = self.dd_alpha * (self._prev_clean_psd / (self._noise_psd + EPS))
        prior_snr += (1.0 - self.dd_alpha) * np.maximum(post_snr - 1.0, 0.0)
        prior_snr = np.maximum(prior_snr, 1e-4)

        llr = (post_snr * prior_snr / (1.0 + prior_snr)) - np.log1p(prior_snr)
        odds = (self.omlsa_q / (1.0 - self.omlsa_q)) * np.exp(-np.clip(llr, -30.0, 30.0))
        p_speech = 1.0 / (1.0 + odds)

        g_h1 = prior_snr / (1.0 + prior_snr)
        g = np.power(g_h1, p_speech) * np.power(self.omlsa_min_gain, 1.0 - p_speech)

        clean_psd = g * g * mag2
        return g, clean_psd, post_snr
