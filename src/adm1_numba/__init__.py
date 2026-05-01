"""ADM1 model accelerated with Numba and numbalsoda."""

from .defaults import DEFAULT_F0, DEFAULT_U0
from .model import ADM1_lsoda, funcptr
from .simulation import calculate_observables, run_stage

__version__ = "0.1.0"

__all__ = [
    "ADM1_lsoda",
    "DEFAULT_F0",
    "DEFAULT_U0",
    "calculate_observables",
    "funcptr",
    "run_stage",
]
