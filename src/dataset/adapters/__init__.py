"""
Data adapters for different model types.
"""
from .grud_adapter import GRUDAdapter
from .hipatch_adapter import HiPatchAdapter
from .kedgn_adapter import KEDGNAdapter
from .dbvt_adapter import DBVTAdapter
from .mtm_adapter import MTMAdapter
from .raindrop_adapter import RaindropAdapter
from .sand_adapter import SANDAdapter
from .strats_adapter import StratsAdapter
from .tcn_adapter import TCNAdapter
from .warpformer_adapter import WarpformerAdapter


# Adapter registry mapping model types to adapter classes
ADAPTER_REGISTRY = {
    # STraTS variants
    'strats': StratsAdapter,
    'istrats': StratsAdapter,
    
    # GRU-D adapter
    'grud': GRUDAdapter,
    
    # KEDGN
    'kedgn': KEDGNAdapter,
    
    # Raindrop
    'raindrop': RaindropAdapter,
    
    # HiPatch
    'hipatch': HiPatchAdapter,
    
    # Warpformer
    'warpformer': WarpformerAdapter,
    
    # MTM
    'mtm': MTMAdapter,
    
    # DBVT (Dual-Branch Variable-Temporal Network)
    'dbvt': DBVTAdapter,
    
    # TCN (Temporal Convolutional Network)
    'tcn': TCNAdapter,
    
    # SAND (Simply Attend and Diagnose)
    'sand': SANDAdapter,

    'ours1': DBVTAdapter,
    'ablation_gru_only': DBVTAdapter,
    'ablation_transformer_only': DBVTAdapter,
    'ablation_no_aux_loss': DBVTAdapter,
    
}


def get_adapter(model_type, args, logger, y=None, demo=None):
    adapter_class = ADAPTER_REGISTRY.get(model_type)
    if adapter_class is None:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {list(ADAPTER_REGISTRY.keys())}")
    
    return adapter_class(args, logger, y, demo)


__all__ = [
    'GRUDAdapter',
    'HiPatchAdapter',
    'KEDGNAdapter',
    'DBVTAdapter',
    'MTMAdapter',
    'RaindropAdapter',
    'SANDAdapter',
    'StratsAdapter',
    'TCNAdapter',
    'WarpformerAdapter',
    'ADAPTER_REGISTRY',
    'get_adapter',
]
