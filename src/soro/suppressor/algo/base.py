import numpy as np
from numpy.fft import irfft, rfft
from scipy.signal import lfilter

from ..utils import design_highpass, hann_sqrt

EPS = 1e-8


class BaseSuppressor:
    def __init__(
        self,
        sr: int,
        frame_ms: int = 20,
        ema_alpha: float = 0.96,
        speech_alpha: float = 0.995,
        noise_gate_db: float = 6.0,
        gain_smooth: float = 0.8,
        highpass_hz: float = 0.0,
        vad_snr_db: float = 3.0,
        vad_speech_ratio: float = 0.25,
        vad_hangover_frames: int = 6,
        noise_floor: float = 0.02,
    ):
        self.sr = int(sr)
        self.frame_len = int(sr * frame_ms / 1000)
        self.hop = self.frame_len // 2
        self.win = hann_sqrt(self.frame_len)

        self.ema_alpha = float(ema_alpha)
        self.speech_alpha = float(speech_alpha)
        self.noise_gate_db = float(noise_gate_db)
        self.gain_smooth = float(gain_smooth)
        self.vad_snr_db = float(vad_snr_db)
        self.vad_speech_ratio = float(vad_speech_ratio)
        self.vad_hangover_frames = int(vad_hangover_frames)
        self.noise_floor = float(noise_floor)

        self._prev_input_tail = np.zeros(self.frame_len - self.hop, dtype=np.float32)
        self._ola = np.zeros(self.frame_len, dtype=np.float32)
        self._prev_gain = None
        self._speech_hold = 0

        self._noise_psd = np.ones(self.frame_len // 2 + 1, dtype=np.float32) * 1e-6
        self._prev_clean_psd = np.zeros(self.frame_len // 2 + 1, dtype=np.float32)
        self._prev_post_snr = np.ones(self.frame_len // 2 + 1, dtype=np.float32)

        self.hp = design_highpass(sr, highpass_hz, order=2)

    def _apply_highpass(self, x: np.ndarray):
        if self.hp is None:
            return x
        b, a, zi = self.hp
        y, zf = lfilter(b, a, x, zi=zi * x[0])
        self.hp = (b, a, zf)
        return y

    def _analyze(self, chunk: np.ndarray):
        frame = np.concatenate([self._prev_input_tail, chunk]).astype(np.float32)
        self._prev_input_tail = chunk[-(self.frame_len - self.hop) :].copy()
        frame = self._apply_highpass(frame)

        x = rfft(frame * self.win)
        mag = np.abs(x)
        mag2 = mag * mag
        phase = np.angle(x)
        return mag, mag2, phase

    def _update_noise_psd(self, mag2: np.ndarray):
        bin_snr_db = 10.0 * np.log10((mag2 + EPS) / (self._noise_psd + EPS))
        speech_bins = bin_snr_db > self.vad_snr_db
        speech_ratio = float(np.mean(speech_bins))
        frame_snr_db = 10.0 * np.log10(np.mean((mag2 + EPS) / (self._noise_psd + EPS)) + EPS)
        frame_has_speech = (speech_ratio >= self.vad_speech_ratio) or (frame_snr_db >= self.noise_gate_db)

        if frame_has_speech:
            self._speech_hold = self.vad_hangover_frames
        elif self._speech_hold > 0:
            self._speech_hold -= 1

        if self._speech_hold > 0:
            alpha = np.where(speech_bins, self.speech_alpha, self.ema_alpha)
        else:
            alpha = np.full_like(mag2, self.ema_alpha)

        self._noise_psd = alpha * self._noise_psd + (1.0 - alpha) * mag2
        return speech_bins

    def _smooth_gain(self, gain: np.ndarray):
        gain = np.maximum(gain, self.noise_floor)
        if self._prev_gain is None:
            self._prev_gain = gain
        gain = self.gain_smooth * self._prev_gain + (1.0 - self.gain_smooth) * gain
        self._prev_gain = gain
        return gain

    def _synthesize(self, gain: np.ndarray, mag: np.ndarray, phase: np.ndarray):
        y = irfft(gain * mag * np.exp(1j * phase)).astype(np.float32)
        y *= self.win

        out = self._ola[: self.hop].copy()
        self._ola[:-self.hop] = self._ola[self.hop :]
        self._ola[-self.hop :] = 0.0
        self._ola += y
        return out

    def calibrate_noise(self, chunk: np.ndarray):
        mag, mag2, _ = self._analyze(chunk)
        self._noise_psd = self.ema_alpha * self._noise_psd + (1.0 - self.ema_alpha) * mag2

    def _compute_gain(self, mag: np.ndarray, mag2: np.ndarray):
        raise NotImplementedError

    def process(self, chunk: np.ndarray) -> np.ndarray:
        mag, mag2, phase = self._analyze(chunk)
        self._update_noise_psd(mag2)
        gain, clean_psd, post_snr = self._compute_gain(mag, mag2)
        gain = self._smooth_gain(gain)
        self._prev_clean_psd = clean_psd.astype(np.float32)
        self._prev_post_snr = post_snr.astype(np.float32)
        return self._synthesize(gain, mag, phase)
