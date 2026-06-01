from pathlib import Path

import logging

from langchain_core.tools import tool

logger = logging.getLogger(f"ollama.{__name__}")

@tool
def read_text_file(path: str) -> str:
    """Read a text file (.txt, .csv, .btl, .ros)
    Args:
        path (str): path to text file
    Returns:
        str | dict: Text file contents
    """
    try:
        with open(path, "r") as f:
            contents = f.read()

        return contents
    except Exception as e:
        return {"Error": str(e)}

def _ls_recursive_list(path: str) -> list[str]:
    """Recursively get a list of directories and subdirectories starting from the provided root
    Args:
        path (str): path to index from
    """
    logger.info(f"Listing directories at path: {path}")
    path_obj = Path(path)
    if not path_obj.is_dir():
        raise NotADirectoryError(f"Not a directory: {path_obj}")

    dirs = [f"{str(dir)}" for dir in path_obj.rglob("*") if dir.is_dir()]
    for dir in dirs:
        dirs += _ls_recursive_list(dir)

    return dirs


@tool
def ls_recursive_list(path: str) -> list[str] | dict:
    """Recursively get a list of directories and subdirectories starting from the provided root
    Args:
        path (str): path to index from
    Returns:
        list[str] | dict: List of directories found at path or a dictionary with an error message
    """
    try:
        return  _ls_recursive_list(path)
    except Exception as e:
        return {"Error": str(e)}


@tool
def ls_directories(path: str) -> list[Path] | dict:
    """Get a list of directories at a given path

    Args:
        path (str): Root path to index from

    Returns:
        list | dict: List of libpath.Path directories found at path or a dictionary with an error message
    """

    logger.info(f"Listing directories at path: {path}")
    try :
        path_obj = Path(path)
        if not path_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {path_obj}")

        dirs = [f"{str(dir)}" for dir in path_obj.iterdir() if dir.is_dir()]
        logger.info(f"Directories found {dirs}")
        return dirs
    except Exception as e:
        return {"Error": str(e)}


@tool
def ls_files(path: str) -> list[str] | dict:
    """Get a list of files in a given path
    Args:
        path (str): path to index from

    Returns:
        list[str] | dict: List of files found at path or a dictionary with an error message
    """
    logger.info(f"Listing files at path: {path}")
    try:
        path_obj = Path(path)
        if not path_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {path_obj}")

        dirs = [f"{str(dir.name)}" for dir in path_obj.iterdir() if dir.is_file()]
        logger.info(f"Files found {dirs}")
        return dirs
    except Exception as e:
        return {"Error": str(e)}