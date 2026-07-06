"""Data adapters for different model types."""
from .gru_adapter import GRUAdapter
from .grud_adapter import GRUDAdapter
from .hipatch_adapter import HiPatchAdapter
from .kedgn_adapter import KEDGNAdapter
from .mtm_adapter import MTMAdapter
from .raindrop_adapter import RaindropAdapter
from .sand_adapter import SANDAdapter
from .strats_adapter import StratsAdapter
from .tcn_adapter import TCNAdapter
from .transformer_adapter import TransformerAdapter
from .warpformer_adapter import WarpformerAdapter

ADAPTER_REGISTRY = {
    'gru': GRUAdapter,
    'grud': GRUDAdapter,
    'hipatch': HiPatchAdapter,
    'kedgn': KEDGNAdapter,
    'mtm': MTMAdapter,
    'raindrop': RaindropAdapter,
    'sand': SANDAdapter,
    'strats': StratsAdapter,
    'tcn': TCNAdapter,
    'transformer': TransformerAdapter,
    'warpformer': WarpformerAdapter,
}


def get_adapter(model_type, args, logger, y=None, demo=None):
    adapter_class = ADAPTER_REGISTRY.get(model_type)
    if adapter_class is None:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(ADAPTER_REGISTRY.keys())}")
    return adapter_class(args, logger, y, demo)


__all__ = ['get_adapter']
