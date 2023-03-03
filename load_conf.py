import tomli
from pathlib import Path
import numpy as np
from typing import Tuple
from typing import List


def load_param_names(toml_file_path: Path) -> List[str]:
    """
    Load parameter names from the config file.
    :param toml_file_path: Path to config file
    :return: List of parameter names
    """
    with open(toml_file_path, 'rb') as toml_file:
        toml_dict = tomli.load(toml_file)
    return [mini_dict['name'] for mini_dict in toml_dict['params']]


def load_bounds(toml_file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numpy arrays of upper and lower parameter bounds
    :param toml_file_path: Path to config file
    :return: Array of lower bounds, array of upper bounds, array of test parameters
    """
    with open(toml_file_path, 'rb') as toml_file:
        toml_dict = tomli.load(toml_file)

    upper_bounds = np.array([mini_dict['upper'] for mini_dict in toml_dict['params']])
    lower_bounds = np.array([mini_dict['lower'] for mini_dict in toml_dict['params']])
    test_params = np.array(toml_dict['hand_tuned_params'][0]['parameter_array'])
    return lower_bounds, upper_bounds, test_params


def load_sampler_config(toml_file_path: Path) -> Tuple[int, int]:
    with open(toml_file_path, 'rb') as toml_file:
        toml_dict = tomli.load(toml_file)

    return toml_dict['model']['chains'], toml_dict['model']['iterations'], toml_dict['model']['dim_full'], toml_dict['model']['params_to_use']#, toml_dict['model']['num_steps']


if __name__ == '__main__':
    load_bounds(Path('conf_t4.toml'))