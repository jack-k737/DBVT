"""
Configuration loader utility.
Loads model-specific configs and merges with command-line arguments.
"""
import yaml
from pathlib import Path
from argparse import Namespace
from typing import Optional, Dict, Any


def load_model_config(model_type: str, dataset: str, config_dir: str = None) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        model_type: Model name (e.g., 'strats', 'warpformer')
        dataset: Dataset name (e.g., 'physionet_2012')
        config_dir: Directory containing config files (default: ../configs)
    
    Returns:
        Dictionary of configuration parameters
    """
    if config_dir is None:
        # Default to configs directory at project root
        config_dir = Path(__file__).parent.parent.parent / 'configs'
    else:
        config_dir = Path(config_dir)
    
    config_file = config_dir / f'{model_type}.yaml'
    
    if not config_file.exists():
        print(f"Warning: Config file not found: {config_file}")
        return {}
    
    with open(config_file, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if all_configs is None:
        return {}
    
    # Get dataset-specific config
    if dataset in all_configs:
        return all_configs[dataset]
    else:
        print(f"Warning: No config found for {model_type} on {dataset}")
        return {}


def merge_config_with_args(args: Namespace, config: Dict[str, Any]) -> Namespace:
    """
    Merge configuration file with command-line arguments.
    Priority: config file > command-line args > default values
    
    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary from YAML
    
    Returns:
        Updated args namespace
    """
    # For each config parameter, always use config value (config file has priority)
    for key, config_value in config.items():
        setattr(args, key, config_value)
    
    return args


def apply_config(args: Namespace) -> Namespace:
    """
    Main function to apply configuration to args.
    
    Args:
        args: Parsed command-line arguments with model_type and dataset
    
    Returns:
        Updated args with config values applied
    """
    # Load model config
    config = load_model_config(args.model_type, args.dataset)
    
    # Merge with args (config file takes priority)
    args = merge_config_with_args(args, config)
    
    return args
