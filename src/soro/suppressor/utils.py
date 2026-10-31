import numpy as np
from scipy.signal import butter, lfilter_zi

_EPS = 1e-8


def hann_sqrt(n: int) -> np.ndarray:
    """Sqrt-Hann window for perfect 50% OLA reconstruction."""
    w = np.hanning(n)
    return np.sqrt(w + _EPS)


def design_highpass(sr: int, cutoff_hz: float, order: int = 2):
    if cutoff_hz <= 0:
        return None
    b, a = butter(order, cutoff_hz / (0.5 * sr), btype="highpass")
    zi = lfilter_zi(b, a)
    return (b, a, zi)
