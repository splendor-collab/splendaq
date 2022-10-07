import numpy as np
from glob import glob
from pathlib import Path
from collections import Counter
from itertools import chain
import os
from datetime import datetime
from scipy.signal import correlate
import warnings

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

        for ind in sorted(np.random.choice(
            inds, size=ncounts, replace=False,
        )):
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
        nrows = datashapes_arr[:, 0]

        arr_range = np.arange(len(filelist))

        choicelist = list(
            chain(
                *[[a] * n * nr for a, n, nr in zip(
                    arr_range, nmax_rows, nrows,
                )]
            )
        )

        if nrandoms > len(choicelist):
            warnings.warn(
                f"The desired number of randoms is too high, as there "
                f"{len(choicelist)} possible randoms. Defaulting to this "
                "maximum amount of randoms."
            )
        rows = np.random.choice(
            choicelist,
            size=np.min([nrandoms, len(choicelist)]),
            replace=False,
        )
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
                        eventtime=(
                            epochtimes[:self._maxevtsperdump]
                        ) + (
                            evtinds[:self._maxevtsperdump] / self._fs
                        ),
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

            dumps_left = len(evtinds) / self._maxevtsperdump

            for ii in range(np.ceil(dumps_left).astype(int)):

                nevents = len(evtinds[:self._maxevtsperdump])
                FW = Writer(f"{self._savepath}{savename}_F{dumpnum:04d}.h5")
                FW.write_data(
                    data=traces[:self._maxevtsperdump],
                    eventindex=evtinds[:self._maxevtsperdump],
                    eventnumber=np.arange(nevents) + basenevents,
                    eventtime=(
                        epochtimes[:self._maxevtsperdump]
                    ) + (
                        evtinds[:self._maxevtsperdump] / self._fs
                    ),
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

                dumps_left = len(evtinds) / self._maxevtsperdump

                if ii + 1 != np.ceil(dumps_left).astype(int):
                    evtinds = evtinds[self._maxevtsperdump:]
                    traces = traces[self._maxevtsperdump:]
                    parentsns = parentsns[self._maxevtsperdump:]
                    parentens = parentens[self._maxevtsperdump:]
                    epochtimes = epochtimes[self._maxevtsperdump:]


    def _initialize_filter(self):
        """
        Method for initializing the optimal filter coefficients.

        """

        # calculate the time-domain optimum filter
        phi_freq = np.fft.fft(self._template)/self._psd
        phi_freq[0] = 0 # ensure we do not use DC information
        self._phi = np.fft.ifft(phi_freq).real
        # calculate the normalization of the optimum filter
        self._norm = np.dot(self._phi, self._template)
        
        # calculate the expected energy resolution
        self._resolution = 1 / (self._norm / self._fs)**0.5


    def _filter_traces(self, traces):
        """
        Method for carrying out the optimal filter in time domain via
        correlation.

        """

        alltraces = traces[:, self._tchan]

        # apply the FIR filter to each trace
        filts = np.array(
            [correlate(
                trace, self._phi, mode="same",
            ) / self._norm for trace in alltraces]
        )

        # set the filtered values to zero near the edges, so as not
        # to use the padded values in the analysis also so that the
        # traces that will be saved will be equal to the tracelength
        cut_len = np.max([len(self._phi), self._tracelength])

        filts[:, :cut_len//2] = 0
        filts[:, -(cut_len//2) + (cut_len + 1) % 2:] = 0

        return filts


    @staticmethod
    def _smart_trigger(trace, threshold):
        """
        Method for carrying out a "smart" triggering algorithm, where
        the turn on threshold is specified, and the turn off threshold
        is the smaller of the turn on threshold and 1/e times the
        maximum OF amplitude.

        """

        turn_on_all = (trace > threshold)
        turn_on = np.zeros(len(turn_on_all), dtype=bool)

        ind1 = 0
        ind_list = []

        searching = True

        while searching:

            turn_on[:] = False
            turn_on[ind1:] = turn_on_all[ind1:]

            if np.all(turn_on==0):
                searching = False
                break

            ind_on = np.argmax(turn_on)
            ind_off_init = np.argmax(np.diff(turn_on.astype(np.int8)) < 0) + 1
            max_amp = np.max(trace[ind_on:ind_off_init])

            if max_amp / np.e < threshold:
                turn_off = (trace > max_amp / np.e)
                turn_off[:ind_on] = 0
                ind0 = np.argmax(turn_off)
                ind1 = np.argmax(np.diff(turn_off.astype(np.int8)) < 0) + 1
            else:
                ind0 = ind_on
                ind1 = ind_off_init

            ind_list.append([ind0, ind1])

        if len(ind_list)==0:
            return []

        return np.vstack(ind_list)


    def acquire_pulses(self, template, psd, threshold, tchan):
        """
        Method to carry out the offline triggering algorithm based on
        the OF formalism in time domain. Only trigeers on one specified
        channel.

        Parameters
        ----------
        template : ndarray
            The amplitude-normalized signal template in time domain.
        psd : ndarray
            The two-sided power spectral density describing the noise
            environment, to be used with the OF.
        threshold : float
            The trigger threshold to set, in units of number of
            expected baseline resolution, e.g. 10 corresponds to a
            10-sigma threshold. If positive, it is assumed that events
            with amplitudes above this value will be extracted. If
            negative, then events with amplitudes below this value
            will be extracted.
        tchan : int
            The channel, designated by array index, to set a threshold
            on and extract events with amplitudes above the threshold.
        
        """
        self._template = template
        self._psd = psd
        self._nthreshold = threshold
        self._tchan = tchan

        self._initialize_filter()
        self._threshold = self._nthreshold * self._resolution

        posthreshold = True if self._nthreshold > 0 else False

        savename = "trigger_" + self._start.strftime("%Y%m%d_%H%M%S")
        seriesnumber = int(self._start.strftime("%y%m%d%H%M%S"))

        filelist = sorted(glob(f"{self._contdatadir}/*.h5"))

        evtinds_list = []
        triginds_list = []
        evtamps_list = []
        traces_list = []
        parentsn_list = []
        parenten_list = []
        epochtime_list = []

        evt_counter = 0
        dumpnum = 1
        basenevents = 0


        for filename in filelist:

            FR = Reader(filename)
            data, metadata = FR.get_data(include_metadata=True)
            parentsn = metadata['parentseriesnumber'][0]
            parenten = metadata['parenteventnumber'][0]
            epochtime_start = metadata['eventtime'][0]
            
            filtered = self._filter_traces(data)

            for kk, filt in enumerate(filtered):

                if posthreshold:
                    ranges = EventBuilder._smart_trigger(
                        filt, self._threshold,
                    )
                else:
                    ranges = EventBuilder._smart_trigger(
                        -filt, -self._threshold,
                    )

                if len(ranges)==0:
                    break

                for ind0, ind1 in zip(ranges[:, 0], ranges[:, 1]):
                    indmax = ind0 + np.argmax(filt[ind0:ind1])
                    evtinds_list.append([indmax - self._tracelength//2])
                    triginds_list.append([indmax])
                    evtamps_list.append([filt[indmax]])
                    trace_save_start = indmax - self._tracelength//2
                    trace_save_end = indmax + self._tracelength//2
                    traces_list.append(
                        data[[kk], :, trace_save_start:trace_save_end]
                    )

                evt_counter += len(ranges)

                parentsn_list.append(parentsn * np.ones(len(ranges)))
                parenten_list.append(parenten * np.ones(len(ranges)))
                epochtime_list.append(epochtime_start * np.ones(len(ranges)))

            if evt_counter >= self._maxevtsperdump:

                evtinds = np.concatenate(evtinds_list)
                triginds = np.concatenate(triginds_list)
                evtamps = np.concatenate(evtamps_list)
                traces = np.concatenate(traces_list)
                parentsns = np.concatenate(parentsn_list)
                parentens = np.concatenate(parenten_list)
                epochtimes = np.concatenate(epochtime_list)

                del evtinds_list
                del triginds_list
                del evtamps_list
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
                        eventtime=(
                            epochtimes[:self._maxevtsperdump]
                        ) + (
                            evtinds[:self._maxevtsperdump] / self._fs
                        ),
                        seriesnumber=[seriesnumber] * nevents,
                        dumpnumber=[dumpnum] * nevents,
                        triggertime=(
                            epochtimes[:self._maxevtsperdump]
                        ) + (
                            triginds[:self._maxevtsperdump] / self._fs
                        ),
                        triggertype=np.ones(nevents, dtype=int),
                        triggeramp=evtamps[:self._maxevtsperdump],
                        parentseriesnumber=parentsns[:self._maxevtsperdump],
                        parenteventnumber=parentens[:self._maxevtsperdump],
                        datashape=traces[:self._maxevtsperdump].shape,
                        fs=self._fs,
                        channels=metadata['channels'],
                        comment=f'trigger threshold: {self._nthreshold}',
                        template=self._template,
                        psd=self._psd,
                    )
                    dumpnum += 1
                    basenevents += nevents

                    evtinds = evtinds[self._maxevtsperdump:]
                    triginds = triginds[self._maxevtsperdump:]
                    evtamps = evtamps[self._maxevtsperdump:]
                    traces = traces[self._maxevtsperdump:]
                    parentsns = parentsns[self._maxevtsperdump:]
                    parentens = parentens[self._maxevtsperdump:]
                    epochtimes = epochtimes[self._maxevtsperdump:]

                if len(evtinds) > 0:
                    evtinds_list = [evtinds]
                    triginds_list = [triginds]
                    evtamps_list = [evtamps]
                    traces_list = [traces]
                    parentsn_list = [parentsns]
                    parenten_list = [parentens]
                    epochtime_list = [epochtimes]
                    evt_counter = len(evtinds)
                else:
                    evtinds_list = []
                    triginds_list = []
                    evtamps_list = []
                    traces_list = []
                    parentsn_list = []
                    parenten_list = []
                    epochtime_list = []
                    evt_counter = 0

        # clean up the remaining events
        if evt_counter > 0:

            evtinds = np.concatenate(evtinds_list)
            triginds = np.concatenate(triginds_list)
            evtamps = np.concatenate(evtamps_list)
            traces = np.concatenate(traces_list)
            parentsns = np.concatenate(parentsn_list)
            parentens = np.concatenate(parenten_list)
            epochtimes = np.concatenate(epochtime_list)

            del evtinds_list
            del triginds_list
            del evtamps_list
            del traces_list
            del parentsn_list
            del parenten_list
            del epochtime_list

            dumps_left = len(evtinds) / self._maxevtsperdump

            for ii in range(np.ceil(dumps_left).astype(int)):

                nevents = len(evtinds[:self._maxevtsperdump])
                FW = Writer(f"{self._savepath}{savename}_F{dumpnum:04d}.h5")
                FW.write_data(
                    data=traces[:self._maxevtsperdump],
                    eventindex=evtinds[:self._maxevtsperdump],
                    eventnumber=np.arange(nevents) + basenevents,
                    eventtime=(
                        epochtimes[:self._maxevtsperdump]
                    ) + (
                        evtinds[:self._maxevtsperdump] / self._fs
                    ),
                    seriesnumber=[seriesnumber] * nevents,
                    dumpnumber=[dumpnum] * nevents,
                    triggertime=(
                        epochtimes[:self._maxevtsperdump]
                    ) + (
                        triginds[:self._maxevtsperdump] / self._fs
                    ),
                    triggertype=np.ones(nevents, dtype=int),
                    triggeramp=evtamps[:self._maxevtsperdump],
                    parentseriesnumber=parentsns[:self._maxevtsperdump],
                    parenteventnumber=parentens[:self._maxevtsperdump],
                    datashape=traces[:self._maxevtsperdump].shape,
                    fs=self._fs,
                    channels=metadata['channels'],
                    comment=f'trigger threshold: {self._nthreshold}',
                    template=self._template,
                    psd=self._psd,
                )
                dumpnum += 1
                basenevents += nevents

                if ii + 1 != np.ceil(dumps_left).astype(int):
                    evtinds = evtinds[self._maxevtsperdump:]
                    triginds = triginds[self._maxevtsperdump:]
                    evtamps = evtamps[self._maxevtsperdump:]
                    traces = traces[self._maxevtsperdump:]
                    parentsns = parentsns[self._maxevtsperdump:]
                    parentens = parentens[self._maxevtsperdump:]
                    epochtimes = epochtimes[self._maxevtsperdump:]

