import os
import tempfile
from typing import BinaryIO


def ensure_data_dir(path: str = "./data") -> None:
    """Ensure the data directory exists."""
    os.makedirs(path, exist_ok=True)


def save_uploaded_file(uploaded_file, suffix: str = ".pdf") -> str:
    """
    Save an uploaded file to a temporary file and return its path.
    Args:
        uploaded_file: The uploaded file object (must support getbuffer()).
        suffix (str): The file extension for the temp file.
    Returns:
        str: The path to the saved temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def cleanup_file(file_path: str) -> None:
    """
    Remove a file if it exists.
    Args:
        file_path (str): The path to the file to remove.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        # Log the error if needed
        pass
