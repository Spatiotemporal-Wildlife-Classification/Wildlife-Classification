"""This file has a single function, to provide the absolute path to the root directory of the project.
    This massively helps development due to the extensive use of relative file paths from the project code,
    and this file providing the single path to the project."""
import sys


def root_dir() -> str:
    """This method provides the absolute path to the root of the project.
    Returns:
        (str): The absolute path to the project root.
    """
    return sys.path[1]
