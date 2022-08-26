import numpy as np
from glob import glob
from pathlib import Path
from collections import Counter
from itertools import chain
import os
from datetime import datetime

from splendaq.io import Reader, Writer


__all__ = [
    'EventBuilder',
]

def rand_sections(x, n, l):
    """
    Return random, non-overlapping sections of an n-dimensional array.
    For greater-than-1 dimensional arrays, the function finds random
    sections along the last axis.

    Parameters
    ----------
    x : ndarray
        n dimensional array to choose sections from.
    n : int
        Number of sections to choose.
    l : int
        Length in bins of sections.

    Returns
    -------
    evtinds : ndarray
        Array of the corresponding event indices for each section.
    res : ndarray
        Array of the n sections of x, each with length l.

    """

    t = np.arange(x.shape[0]) * x.shape[-1]

    tup = ((n,), x.shape[1:-1], (l,))
    sz = sum(tup, ())

    res = np.zeros(sz)
    evtinds = np.zeros(n, dtype=int)
    j = 0

    nmax = int(x.shape[-1] / l)

    if x.shape[0] * nmax < n:
        raise ValueError(
            "Either n or l is too large, trying to find more "
            "random sections than are possible."
        )

    choicelist = list(range(len(x))) * nmax
    rows = np.random.choice(choicelist, size=n, replace=False)
    counts = Counter(rows)


    for key in counts:
        offset = 0
        ncounts = counts[key]
        inds = np.arange(x.shape[-1] - (l - 1) * ncounts)

        for ind in sorted(np.random.choice(inds, size=ncounts, replace=False)):
            ind += offset
            res[j] = x[key, ..., ind:ind + l]
            evtinds[j] = t[key] + ind + l//2
            j += 1
            offset += l - 1
    
    return evtinds, res


class EventBuilder(object):
    """
    Class to build events from continuous data, for both randomly
    triggered events and threshold triggered events.

    """

    def __init__(self, contdatadir, savepath, tracelength,
                 maxevtsperdump=500):
        """
        Initialization of the EventBuilder class.

        Parameters
        ----------
        contdatadir : str
            Absolute path to the directory containing the continuous
            data.
        savepath : str
            Absolute path to the directory to save the built events.
        tracelength : int
            The length of each event built in units of bins.
        maxevtsperdump : int, optional
            The maximum number of events to save to each file created.
        """

        self._contdatadir = contdatadir
        self._tracelength = tracelength
        self._maxevtsperdump = maxevtsperdump
        self._savepath = f"{os.path.abspath(savepath)}{os.sep}"

        self._start = datetime.now()

        filelist = sorted(glob(f"{self._contdatadir}/*.h5"))
        FR = Reader(filelist[0])
        metadata = FR.get_metadata()
        self._fs = metadata['fs']

    def acquire_randoms(self, nrandoms):
        """
        Method for acquiring randomly triggered events and
        saving them to the specified file location.

        Parameters
        ----------
        nrandoms : int
            The number of randoms to acquire from the continuous
            data.

        """

        savename = "randoms_" + self._start.strftime("%Y%m%d_%H%M%S")
        seriesnumber = int(self._start.strftime("%y%m%d%H%M%S"))

        filelist = sorted(glob(f"{self._contdatadir}/*.h5"))
        datashapes = []
        for filename in filelist:
            FR = Reader(filename)
            metadata = FR.get_metadata()
            datashapes.append(metadata['datashape'])
        datashapes_arr = np.vstack(datashapes)

        nmax_rows = datashapes_arr[..., -1] // self._tracelength
        arr_range = np.arange(len(filelist))

        choicelist = list(
            chain(*[[a] * n for n, a in zip(nmax_rows, arr_range)])
        )
        rows = np.random.choice(choicelist, size=nrandoms, replace=False)
        counts = Counter(rows)

        evtinds_list = []
        traces_list = []
        parentsn_list = []
        parenten_list = []
        epochtime_list = []

        evt_counter = 0
        dumpnum = 1
        basenevents = 0

        for key in sorted(counts.keys()):
            
            FR = Reader(filelist[key])
            data, metadata = FR.get_data(include_metadata=True)
            parentsn = metadata['parentseriesnumber'][0]
            parenten = metadata['parenteventnumber'][0]
            epochtime_start = metadata['eventtime'][0]

            ei, tr = rand_sections(data, counts[key], self._tracelength)

            evt_counter += len(ei)

            evtinds_list.append(ei)
            traces_list.append(tr)
            parentsn_list.append(parentsn * np.ones(len(ei)))
            parenten_list.append(parenten * np.ones(len(ei)))
            epochtime_list.append(epochtime_start * np.ones(len(ei)))

            if evt_counter >= self._maxevtsperdump:

                evtinds = np.concatenate(evtinds_list)
                traces = np.vstack(traces_list)
                parentsns = np.concatenate(parentsn_list)
                parentens = np.concatenate(parenten_list)
                epochtimes = np.concatenate(epochtime_list)

                del evtinds_list
                del traces_list
                del parentsn_list
                del parenten_list
                del epochtime_list

                for ii in range(len(evtinds) // self._maxevtsperdump):

                    nevents = len(evtinds[:self._maxevtsperdump])
                    FW = Writer(
                        f"{self._savepath}{savename}_F{dumpnum:04d}.h5",
                    )
                    FW.write_data(
                        data=traces[:self._maxevtsperdump],
                        eventindex=evtinds[:self._maxevtsperdump],
                        eventnumber=np.arange(nevents) + basenevents,
                        eventtime=epochtimes[:self._maxevtsperdump] + evtinds[:self._maxevtsperdump] / metadata['fs'],
                        seriesnumber=[seriesnumber] * nevents,
                        dumpnumber=[dumpnum] * nevents,
                        triggertime=np.zeros(nevents),
                        triggertype=np.zeros(nevents, dtype=int),
                        triggeramp=np.zeros(nevents),
                        parentseriesnumber=parentsns[:self._maxevtsperdump],
                        parenteventnumber=parentens[:self._maxevtsperdump],
                        datashape=traces[:self._maxevtsperdump].shape,
                        fs=self._fs,
                        channels=metadata['channels'],
                        comment='randoms',
                    )
                    dumpnum += 1
                    basenevents += nevents

                    evtinds = evtinds[self._maxevtsperdump:]
                    traces = traces[self._maxevtsperdump:]
                    parentsns = parentsns[self._maxevtsperdump:]
                    parentens = parentens[self._maxevtsperdump:]
                    epochtimes = epochtimes[self._maxevtsperdump:]

                if len(evtinds) > 0:
                    evtinds_list = [evtinds]
                    traces_list = [traces]
                    parentsn_list = [parentsns]
                    parenten_list = [parentens]
                    epochtime_list = [epochtimes]
                    evt_counter = len(evtinds)
                else:
                    evtinds_list = []
                    traces_list = []
                    parentsn_list = []
                    parenten_list = []
                    epochtime_list = []
                    evt_counter = 0

        # clean up the remaining events
        if evt_counter > 0:

            evtinds = np.concatenate(evtinds_list)
            traces = np.vstack(traces_list)
            parentsns = np.concatenate(parentsn_list)
            parentens = np.concatenate(parenten_list)
            epochtimes = np.concatenate(epochtime_list)

            del evtinds_list
            del traces_list
            del parentsn_list
            del parenten_list
            del epochtime_list

            for ii in range(np.ceil(len(evtinds) / self._maxevtsperdump).astype(int)):
                
                nevents = len(evtinds[:self._maxevtsperdump])
                FW = Writer(f"{self._savepath}{savename}_F{dumpnum:04d}.h5")
                FW.write_data(
                    data=traces[:self._maxevtsperdump],
                    eventindex=evtinds[:self._maxevtsperdump],
                    eventnumber=np.arange(nevents) + basenevents,
                    eventtime=epochtimes[:self._maxevtsperdump] + evtinds[:self._maxevtsperdump] / metadata['fs'],
                    seriesnumber=[seriesnumber] * nevents,
                    dumpnumber=[dumpnum] * nevents,
                    triggertime=np.zeros(nevents),
                    triggertype=np.zeros(nevents, dtype=int),
                    triggeramp=np.zeros(nevents),
                    parentseriesnumber=parentsns[:self._maxevtsperdump],
                    parenteventnumber=parentens[:self._maxevtsperdump],
                    datashape=traces[:self._maxevtsperdump].shape,
                    fs=self._fs,
                    channels=metadata['channels'],
                    comment=None,
                )
                dumpnum += 1
                basenevents += nevents

                if ii + 1 != np.ceil(len(evtinds) / self._maxevtsperdump).astype(int):
                    evtinds = evtinds[self._maxevtsperdump:]
                    traces = traces[self._maxevtsperdump:]
                    parentsns = parentsns[self._maxevtsperdump:]
                    parentens = parentens[self._maxevtsperdump:]
                    epochtimes = epochtimes[self._maxevtsperdump:]
