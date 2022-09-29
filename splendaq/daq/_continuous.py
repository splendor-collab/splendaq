import contextlib
import time
import os
import numpy as np

from moku.instruments import Datalogger


__all__ = [
    "connect_logger",
    "log_data",
]


@contextlib.contextmanager
def connect_logger(ip_address, force_connect=False):
    """
    Context manager for connecting to the Moku Datalogger module.

    Parameters
    ----------
    ip_address : str
        The IP address for the Moku to connect to.
    force_connect : bool, optional
        If the Moku is being used by some other source, setting to
        True will supersede that source and connect. Default is False.

    Returns
    -------
    DL : object
        The Datalogger object connected to the specified Moku.

    """

    DL = Datalogger(ip_address, force_connect=force_connect)
    try:
        yield DL
    except Exception as e:
        raise e
    finally:
        DL.relinquish_ownership()


def log_data(ip_address, fs, duration, channels, savepath,
             force_connect=False, comment='', file_name_prefix='',
             max_duration_per_file=60):
    """
    Function for logging continuous data on a Moku for a specified
    duration and channels.

    Parameters
    ----------
    ip_address : str
        The IP address for the Moku to connect to.
    fs : float
        The digitization rate of the data that will be logged.
    duration : float
        The amount of time in seconds that will be logged. Will not be
        exactly followed due to time to tell the Moku to stop, and is
        generally a lower bound per file.
    channels : int, list of ints
        Which channels to log, should be an integer from 1, 2, 3, or 4.
        If multiple channels are to be logged, a list can be passed.
    savepath : str
        The local directory in which to save the logged file.
    force_connect : bool, optional
        If the Moku is being used by some other source, setting to
        True will supersede that source and connect. Default is False.
    comment : str, optional
        An optional comment to store in the data, generally a
        description of what the data is.
    file_name_prefix : str, optional
        An optional prefix for the filename. the default is given
        by the Moku: MokuDataLoggerData.
    max_duration_per_file : float, optional
        The maximum amount of time per file in seconds. Default is
        60 seconds.

    """

    with connect_logger(ip_address, force_connect=force_connect) as DL:
        chan_list = [1, 2, 3, 4]
        if np.isscalar(channels):
            channels = [channels]

        for channel in channels:
            DL.set_frontend(
                channel=channel,
                impedance='1MOhm',
                coupling="AC",
                range="40Vpp",
            )

        disable_chans = [chan not in channels for chan in chan_list]
        for chan, disable in zip(chan_list, disable_chans):
            if disable:
                DL.disable_channel(chan)
        DL.set_samplerate(fs)
        DL.set_acquisition_mode(mode='Normal')

        filenames = []
        nfiles = int(np.ceil(duration / max_duration_per_file))

        for ii in range(nfiles):
            logfile = DL.start_logging(
                duration=duration,
                comments=comment,
                file_name_prefix=file_name_prefix,
            )
            # Track progress percentage of the data logging session
            is_logging = True
            while is_logging:
                # Wait for the logging session to progress by sleeping 0.5sec
                time.sleep(0.5)
                # Get current progress percentage and print it out
                progress = DL.logging_progress()
                remaining_time = progress['time_remaining']
                is_logging = remaining_time >= 0

            filenames.append(logfile['file_name'])

        for fname in filenames:
            DL.download(
                "ssd",
                fname,
                os.path.abspath(savepath + os.sep + fname),
            )

