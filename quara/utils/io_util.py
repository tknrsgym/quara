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


def check_positive_number(target: float, parameter_name: str) -> None:
    """check if ``target`` is positive number.
    
    Parameters
    ----------
    target : int
        the number to check.
    parameter_name : str
        the parameter name of the number.
        this is used for error message.
    
    Raises
    ------
    ValueError
        ``target`` is not positive number.
    """
    # check data
    if target <= 0:
        raise ValueError(
            f"Invalid value range '{parameter_name}'. expected>0, actual={target}"
        )


def check_nonnegative_number(target: float, parameter_name: str) -> None:
    """check if ``target`` is nonnegative number.
    
    Parameters
    ----------
    target : int
        the number to check.
    parameter_name : str
        the parameter name of the number.
        this is used for error message.
    
    Raises
    ------
    ValueError
        ``target`` is not nonnegative number.
    """
    # check data
    if target < 0:
        raise ValueError(
            f"Invalid value range '{parameter_name}'. expected>=0, actual={target}"
        )
