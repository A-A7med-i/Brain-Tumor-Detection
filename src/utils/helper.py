import pickle
import yaml
import pandas as pd
from pathlib import Path
from typing import Union, Any, Dict


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def create_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a dictionary.

    Args:
        data (Dict[str, Any]): Dictionary containing data to be converted into a DataFrame.

    Returns:
        pd.DataFrame: Created DataFrame.

    Raises:
        ValueError: If the data cannot be converted into a DataFrame.
    """
    try:
        return pd.DataFrame(data)

    except ValueError as e:
        raise ValueError(f"Failed to create DataFrame: {e}")


def save_data(data: Any, data_path: Union[str, Path]) -> None:
    """
    Save data to a file using pickle.

    Args:
        data (Any): Data to be saved.
        data_path (Union[str, Path]): Path to the file where data will be saved.

    Raises:
        IOError: If there is an error writing to the file.
    """
    data_path = Path(data_path)

    try:
        with open(data_path, "wb") as file:
            pickle.dump(data, file)
    except IOError as e:
        raise IOError(f"Failed to save data to {data_path}: {e}")
