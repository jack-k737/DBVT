"""
Configuration loader utility.
Loads model-specific configs and merges with command-line arguments.
"""
import yaml
from pathlib import Path
from argparse import Namespace
from typing import Dict, Any


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

    if dataset in all_configs:
        return all_configs[dataset]

    print(f"Warning: No config found for {model_type} on {dataset}")
    return {}


def merge_config_with_args(args: Namespace, config: Dict[str, Any]) -> Namespace:
    """
    Merge configuration file with command-line arguments.
    Priority: config file > command-line args > default values
    """
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
    config_model_type = getattr(args, 'config_model_type', args.model_type)

    # Treat xxx / xxx+ / xxx++ as distinct variants and prefer the exact config file.
    config = load_model_config(config_model_type, args.dataset)
    if not config and config_model_type != args.model_type:
        config = load_model_config(args.model_type, args.dataset)

    args = merge_config_with_args(args, config)
    return args
