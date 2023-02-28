import numpy as np
import os
import time

from moku.instruments import Oscilloscope as Osc


__all__ = [
    "Oscilloscope",
]


class Oscilloscope(object):
    """
    Class for managing the Oscilloscope object from the Moku package
    to take data with various settings safely.

    Attributes
    ----------
    Osc : moku.instruments.Oscilloscope
        The wrapped Oscilloscope object that interfaces with the Moku.

    """

    def __init__(self, ip_address, force_connect=False,
                 acquisition_mode="Precision"):
        """
        Initialization of the Oscilloscope class.

        Parameters
        ----------
        ip_address : str
            The IP address of the Moku to connect to. If connecting via
            IPv6 (e.g. via USB-C), ensure to enclose the address with
            square brackets, e.g. "[your_ip_address]".
        force_connect : bool, optional
            Take ownership of the Moku even if it is being used by
            someone else. Default is False.
        acquisition_mode : str, optional
            Changes acquisition mode between 'Normal' and 'Precision'.
            Precision mode is also known as decimation, it samples at
            the full rate and averages excess data points to improve
            precision. Normal mode works by direct down sampling,
            throwing away extra data points. Default is 'Precision'.

        """

        self.Osc = Osc(
            ip_address, force_connect=force_connect, session_trust_env=False,
        )
        self.Osc.set_acquisition_mode(acquisition_mode)

        self._device = self.Osc.describe()['hardware']
        if self._device not in ['Moku:Pro', 'Moku:Go']:
            raise ValueError(
                "Unrecognized device, is not a Moku:Go or Moku:Pro."
            )

    def __enter__(self):
        """Returns self when entered via with."""
        return self

    def __exit__(self, type, value, traceback):
        """
        Always run relinquish ownership and turn off inputs and outputs
        when exiting.

        """

        if self._device == "Moku:Pro":
            chan_list = [1, 2, 3, 4]
        elif self._device == "Moku:Go":
            chan_list = [1, 2]

        for chan in chan_list:
            self.Osc.generate_waveform(chan, 'Off') # disable outputs
            self.Osc.disable_input(chan) # disable inputs

        self.Osc.relinquish_ownership()

    def set_input_channels(self, channels, impedance="1MOhm", coupling="DC",
                           vrange=None):
        """
        Set which channels to save data from and which settings to use.

        Parameters
        ----------
        channels : int, list of int
            Which input channels to log data from. Can be a single
            value for one channel, or a list of the channels. For the
            Moku:Pro, valid channel numbers are 1, 2, 3, and 4. For the
            Moku:Go, valid channel numbers are 1 and 2.
        impedance : str, list of str, optional
            The output impedance to use for each channel. For the
            Moku:Pro, options are '1MOhm' and '50Ohm'. For the Moku:Go,
            the only option is '1MOhm'. Can pass a list of the same
            length as channels if different impedances are desired for
            a Moku:Pro. Default is '1MOhm' for all channels.
        coupling : str, list of str, optional
            The coupling to use for each channel. Options are 'DC' and
            'AC'. Can pass a list of the same length as channels if
            different couplings are desired. Default is 'DC' for all
            channels.
        vrange : str, list of str, optional
            The voltage range to use for each channel. For the
            Moku:Pro, options are '400mVpp', '4Vpp' and '40Vpp'. For
            the Moku:Go, options are '10Vpp' and '50Vpp'. Can pass a
            list of the same length as channels if different voltage
            ranges are desired for specific channels. Default is the
            smallest allowed value for the device for all channels
            (i.e. '400mVpp' for the Moku:Pro and '10Vpp' for the
            Moku:Go).

        """

        if self._device == "Moku:Pro":
            chan_list = [1, 2, 3, 4]
            vrange = '400mVpp' if vrange is None else vrange
        elif self._device == "Moku:Go":
            chan_list = [1, 2]
            vrange = '10Vpp' if vrange is None else vrange

        if np.isscalar(channels):
            channels = [channels]
        if np.isscalar(impedance):
            impedance = [impedance] * len(channels)
        if np.isscalar(coupling):
            coupling = [coupling] * len(channels)
        if np.isscalar(vrange):
            vrange = [vrange] * len(channels)

        disable_chans = [chan not in channels for chan in chan_list]
        for chan, disable in zip(chan_list, disable_chans):
            if disable:
                self.Osc.disable_input(chan)
            else:
                ind = channels.index(chan)
                self.Osc.set_frontend(
                    channel=chan,
                    impedance=impedance[ind],
                    coupling=coupling[ind],
                    range=vrange[ind],
                )

    def set_output_channels(self, channels, waveformtype='DC', load="1MOhm",
                            dc_level=0):
        """
        Method to turn on an output channel and generate the speicified
        waveform. Use the various self.*_settings() methods to
        generate the various valid settings. Can only set one channel
        at a time.

        Parameters
        ----------
        channel : int, list of int
            The output channel to generate a waveform on. With a
            Moku:Pro, this must be an integer of 1, 2, 3 or 4. With a
            Moku:Go, this must be an integer of 1 or 2.
        waveformtype : str, list of str, optional
            The waveform type to generate, only supports 'DC' waveforms
            at the moment. Can also be set to 'Off' to turn the channel
            off. Default is 'DC'.
        load : str, list of str, optional
            The load impedance to use for the output channel. For a
            Moku:Pro, this must be one of '1MOhm' or '50Ohm'. For a
            Moku:Go, this can only be '1MOhm'. Default is '1MOhm'.
        dc_level : float, list of float, optional
            The dictionary containing all of the settings needed for
            the specified waveform type. Default is 0.

        """

        if np.isscalar(channels):
            channels = [channels]
        if np.isscalar(waveformtype):
            waveformtype = [waveformtype] * len(channels)
        if np.isscalar(load):
            load = [load] * len(channels)
        if np.isscalar(dc_level):
            dc_level = [dc_level] * len(channels)

        for ind, chan in enumerate(channels):
            if self._device == "Moku:Pro":
                self.Osc.set_output_load(chan, load[ind])
            self.Osc.generate_waveform(
                channel=chan,
                type=waveformtype[ind],
                dc_level=dc_level[ind],
            )

    def get_data(self, sources, duration):
        """
        Method to get data after setting up the desired configuration.
        Returns a dictionary with the time domain values.

        Parameters
        ----------
        sources : str, list of str
            The sources which to read out in order of increasing
            channel number. Can be some combination of "Input1",
            "Input2", "Output1", "Output2" for the Moku:Go, with
            "Input3", "Input4", "Output3", and "Output4" also available
            for the Moku:Pro.
        duration : float
            The total time in seconds to collect data with the current
            configuration.

        Returns
        -------
        data : dict
            Dictionary with the time-domain data of the sources
            specified. The keys of the sources are labeled as
            "ch1", "ch2", etc. The digitization rate of the data is
            always 1024 / `duration`.

        """

        for ii, s in enumerate(sources):
            self.Osc.set_source(ii + 1, s)

        self.Osc.set_timebase(0, duration)
        time.sleep(duration)

        data = self.Osc.get_data()

        return data

