from pathlib import Path


def check_file_extension(path: str) -> None:
    """check if the file extension is `csv`.

    Parameters
    ----------
    path : str
        the file path to check.

    Raises
    ------
    ValueError
        the file extension is not `csv`.
    """
    extension = Path(path).suffix
    target_extensions = [".csv"]
    if extension not in target_extensions:
        raise ValueError(
            f"Invalid file extension in '{path}'. expected={target_extensions}, actual={extension}"
        )
