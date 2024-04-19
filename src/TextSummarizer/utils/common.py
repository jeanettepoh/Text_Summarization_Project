import os
import yaml
from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from TextSummarizer.logging import logger

@ensure_annotations
def read_yaml(yaml_filepath: Path) -> ConfigBox:
    """
    Parses yaml file and returns its contents

    Args:
        yaml_filepath (Path): Path to yaml file
    
    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(yaml_filepath, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {yaml_filepath} loaded sucessfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list):
    """
    Creates directories if they don't exist

    Args:
        path_to_directories (list): List of directories to be created
    """
    for directory in path_to_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Creating directory: {directory}")


@ensure_annotations
def get_file_size(path: Path) -> str:
    """
    Returns file size in KB

    Args:
        path (Path): Path to file

    Returns:
        str: Size of file in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"{size_in_kb} KB"