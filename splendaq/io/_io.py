import subprocess
from os import path


__all__ = [
    "convert_li",
]

THIS_DIRECTORY = path.abspath(path.dirname(__file__))

def convert_li(file, filetype='npy'):
    """
    Python wrapper for liconvert executable for converting `.li` files
    to `.csv`, `.mat`, or `.npy`.

    Parameters
    ----------
    file : str
        The absolute file path and name to the `.li` file to convert.
    filetype : str, optional
        The file type to convert `file` to, can be "csv", "npy", or
        "mat". Default is "npy".

    """

    return subprocess.call(
        [path.join(
            THIS_DIRECTORY,
            '_liconvert/liconvert',
        ), f"--{filetype}", file]
    )
