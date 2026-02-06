"""
Data module for dataset loading and preprocessing.
"""
from .dataset import Dataset
from .base_adapter import BaseDataAdapter
from .adapters import get_adapter, ADAPTER_REGISTRY

__all__ = [
    'Dataset',
    'BaseDataAdapter',
    'get_adapter',
    'ADAPTER_REGISTRY',
]

