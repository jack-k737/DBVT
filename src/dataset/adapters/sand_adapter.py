import sys
from pathlib import Path

try:
    from .tcn_adapter import TCNAdapter
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from tcn_adapter import TCNAdapter


class SANDAdapter(TCNAdapter):
    pass
