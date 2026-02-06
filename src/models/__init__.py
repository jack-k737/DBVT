"""
Models package for time series classification.
"""

from .grud import GRUD_TS
from .hipatch import HiPatch
from .kedgn import KEDGN
from .dbvt import DBVT
from .mtm import MTM
from .raindrop import Raindrop
from .sand import SAND
from .strats import Strats
from .tcn import TCN
from .warpformer import Warpformer

# Backward compatibility
MedBiVT = DBVT

__all__ = [
    'GRUD_TS',
    'HiPatch',
    'KEDGN',
    'DBVT',
    'MedBiVT',  # Backward compatibility
    'MTM',
    'Raindrop',
    'SAND',
    'Strats',
    'TCN',
    'Warpformer',
]

