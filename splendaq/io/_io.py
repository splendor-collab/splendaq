import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.io import loadmat
import h5py


__all__ = [
    "Reader",
    "Writer",
    "convert_li_to_h5",
]


THIS_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


class Reader(object):
    """
    File reader for splendaq HDF5 files.

    Attributes
    ----------
    filename : str
        The path to the HDF5 file to open.

    """

    def __init__(self, filename=None):
        """
        Initialization of the HDF5 file reader.

        Parameters
        ----------
        filename : str, optional
            The path to the HDF5 file to open.

        """

        self.filename = filename

    def _open_file(self, include_data, include_metadata, filename=None):
        """
        Hidden method to handle opening data under different settings.

        """

        if filename is not None:
            self.filename = filename
        elif self.filename is None:
            raise ValueError("No filename specified.")

        with h5py.File(self.filename, mode='r') as hf:
            if include_metadata:
                out_dict = dict(hf.attrs)
                for key in hf:
                    if key != 'data':
                        out_dict[key] = np.asarray(hf[key])
            if include_data:
                data = np.asarray(hf['data'])

        if include_data and include_metadata:
            return data, out_dict
        elif include_data and not include_metadata:
            return data
        elif not include_data and include_metadata:
            return out_dict


    def get_data(self, filename=None, include_metadata=False):
        """
        Method to load the data from the specified HDF5 file.

        Parameters
        ----------
        filename : str, optional
            The path to the HDF5 file to open. Can be set here if not
            set when initializing, or if a different file will be read.
        include_metadata : bool, optional
            If True, the metadata will also be returned alongside the
            data in the HDF5 file. Default is False.

        Returns
        -------
        data : ndarray
            The array of trace data saved to the HDF5 file.
        out_dict : dict, optional
            A dictionary containing the corresponding metadata for the
            events.

        """

        return self._open_file(True, include_metadata, filename=filename)

    def get_metadata(self, filename=None):
        """
        Method to load only the metadata from the specified HDF5 file.

        Parameters
        ----------
        filename : str, optional
            The path to the HDF5 file to open. Can be set here if not
            set when initializing, or if a different file will be read.

        Returns
        -------
        out_dict : dict
            A dictionary containing the corresponding metadata for the
            events.

        """

        return self._open_file(False, True, filename=filename)


class Writer(object):
    """
    File writer for splendaq HDF5 files.

    Attributes
    ----------
    filename : str
        The path and filename to save the HDF5 file as.

    """

    def __init__(self, filename=None):
        """
        Initialization of the HDF5 file reader.

        Parameters
        ----------
        filename : str, optional
            The path to the HDF5 file to open.

        """

        self.filename = filename

    def write_data(self, data, filename=None, **metadata):
        """
        Method to save the data to the specified HDF5 file.

        Parameters
        ----------
        data : ndarray
            The ndarray of data to save to the HDF5 file.
        filename : str, optional
            The path to the HDF5 file to open. Can be set here if not
            set when initializing, or if a different file will be read.
        metadata : kwargs
            All of the corresponding metadata for the events in the
            HDF5 file.

        """

        if filename is not None:
            self.filename = filename
        elif self.filename is None:
            raise ValueError("No filename specified.")

        with h5py.File(self.filename, mode='w') as hf:
            hf.create_dataset('data', data=data, compression='gzip')
            for key in metadata:
                if sys.getsizeof(metadata[key]) < 64000:
                    hf.attrs[key] = metadata[key]
                else:
                    hf.create_dataset(
                        key, data=metadata[key], compression='gzip',
                    )
                


def convert_li(file, my_os, filetype='mat'):
    """
    Python wrapper for liconvert executable for converting `.li` files
    to `.csv`, `.mat`, or `.npy`.

    Parameters
    ----------
    file : str
        The absolute file path and name to the `.li` file to convert.
    my_os : str
        The operating system to assume when trying to convert a file.
        Can be 'mac', 'linux', or 'windows'.
    filetype : str, optional
        The file type to convert `file` to, can be "csv", "npy", or
        "mat". Default is "mat".

    Returns
    -------
    subprocess_obj : CompletedProcess
        The completed subprocess, including information on what was
        run and error code.

    Raises
    ------
    ValueError
        If `os` is not one of 'mac', 'linux', or 'windows'.

    """

    if my_os == "mac":
        exe = "liconvert_macos"
    elif my_os == "linux":
        exe = "liconvert_linux"
    elif my_os == "windows":
        exe = "liconvert_windows.exe"
    else:
        raise ValueError("Check docstring for supported OSs")

    return subprocess.run(
        [os.path.join(
            THIS_DIRECTORY,
            f'_liconvert{os.sep}{exe}',
        ), f"--{filetype}", file],
        stdout=False,
    )


def convert_li_to_h5(li_file, my_os):
    """
    Converting a LI file to a splendaq HDF5 file.

    Parameters
    ----------
    li_file : str
        The path to the binary LI file to convert to a splendaq HDF5
        file.
    my_os : str
        The operating system to assume when trying to convert a file.
        Can be 'mac', 'linux', or 'windows'.

    """

    convert_li(li_file, my_os, filetype='mat')
    mat_filename = Path(li_file).with_suffix('.mat')
    mat_file = loadmat(mat_filename, simplify_cells=True)['moku']
    os.remove(mat_filename)

    comment_str = mat_file['comment']
    comment = list(
        filter(None, comment_str.replace('\r\n', '').split('% '))
    )

    kw_timestamp = "Acquired"
    moku_timestamp = [
        s for s in comment if kw_timestamp in s
    ][0][len(kw_timestamp):].replace(' ','')
    iso_timestamp = moku_timestamp[:-2] + ':' + moku_timestamp[-2:]
    begintime = datetime.fromisoformat(iso_timestamp)
    seriesnumber = int(begintime.strftime("%y%m%d%H%M%S"))
    epochtime = begintime.timestamp()

    kw_fs = "Acquisition rate:"
    fs = float([
        s for s in comment if kw_fs in s
    ][0][len(kw_fs):].split('Hz')[0])

    arr = mat_file['data']
    columns = mat_file['legend']

    new_filename = Path(li_file).with_suffix('.h5')
    datashape = arr.T[None, 1:].shape

    FW = Writer(new_filename)
    FW.write_data(
        data=arr.T[None, 1:],
        channels=list(columns[1:]),
        comment=comment,
        fs=fs,
        datashape=datashape,
        eventindex=[0] * datashape[0],
        eventnumber=np.arange(datashape[0]),
        eventtime=[epochtime] * datashape[0],
        seriesnumber=[seriesnumber] * datashape[0],
        dumpnumber=[1] * datashape[0],
        triggertime=[epochtime] * datashape[0],
        triggertype=[0] * datashape[0],
        parentseriesnumber=[seriesnumber] * datashape[0],
        parenteventnumber=np.arange(datashape[0]),
    )

