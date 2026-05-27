from pathlib import Path

import logging
logger = logging.getLogger(f"ollama.{__name__}")

def ls_directories(root_path: str) -> list[str]:
    """Get a list of directories at a given path

    Args:
        root_path (str): Root path to index from

    Returns:
        str: List of directories found at path
    """

    logger.info(f"Listing directories at path: {root_path}")
    path_obj = Path(root_path)
    if not path_obj.is_dir():
        raise NotADirectoryError(f"Not a directory: {path_obj}")

    dirs = [f"./{str(dir).replace("\\", "/")}" for dir in path_obj.iterdir() if dir.is_dir()]
    logger.info(f"Directories found {dirs}")
    return dirs

def ls_files(path: str) -> list[str]:
    """Get a list of files from a given path
    Args:
        path (str): Root path to index from

    Returns:
        list[str]: List of files found at path
    """
    logger.info(f"Listing files at path: {path}")
    path_obj = Path(path)
    if not path_obj.is_dir():
        raise NotADirectoryError(f"Not a directory: {path_obj}")

    dirs = [f"{str(dir.name)}" for dir in path_obj.iterdir() if dir.is_file()]
    logger.info(f"Files found {dirs}")
    return dirs
