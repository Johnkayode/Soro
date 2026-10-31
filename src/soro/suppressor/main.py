import inspect
from enum import Enum
from typing import Dict, Type
from .algo.base import BaseSuppressor
from .algo.dd_weiner import DDWienerSuppressor
from .algo.log_mmse import LogMMSESuppressor
from .algo.omlsa import OMLSASuppressor
from .algo.spectral_subtraction import SpectralSubtractionSuppressor


class SuppressionAlgorithm(str, Enum):
    DDWiener = "dd_weiner"
    LogMMSE = "log_mmse"
    OMLSA = "omlsa"
    SpectralSubtraction = "spectral_subtraction"


suppression_algorithms: Dict[SuppressionAlgorithm, Type[BaseSuppressor]] = {
    SuppressionAlgorithm.DDWiener: DDWienerSuppressor,
    SuppressionAlgorithm.LogMMSE: LogMMSESuppressor,
    SuppressionAlgorithm.OMLSA: OMLSASuppressor,
    SuppressionAlgorithm.SpectralSubtraction: SpectralSubtractionSuppressor,
}


def build_suppressor(algo: SuppressionAlgorithm, **kwargs):
    def _build(cls):
        base_params = set(inspect.signature(BaseSuppressor.__init__).parameters)
        cls_params = set(inspect.signature(cls.__init__).parameters)
        allowed_keys = (base_params | cls_params) - {"self", "args", "kwargs"}
        allowed = {k: v for k, v in kwargs.items() if k in allowed_keys}
        return cls(**allowed)

    return _build(suppression_algorithms[algo])
