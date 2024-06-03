import time
import os
import numpy as np

from moku.instruments import Datalogger


__all__ = [
    "LogData",
]


class LogData(object):
    """
    Class for managing the Datalogger object from the Moku package
    to take data with various settings safely.

    Attributes
    ----------
    DL : moku.instruments.Datalogger
        The wrapped Datalogger object that interfaces with the Moku.

    """

    def __init__(self, ip_address, force_connect=False, fs=None,
                 max_duration_per_file=60,
                 acquisition_mode="Precision"):
        """
        Initialization of the LogData class.

        Parameters
        ----------
        ip_address : str
            The IP address of the Moku to connect to. If connecting via
            IPv6 (e.g. via USB-C), ensure to enclose the address with
            square brackets, e.g. "[your_ip_address]".
        force_connect : bool, optional
            Take ownership of the Moku even if it is being used by
            someone else. Default is False.
        fs : float, optional
            The digitization rate of the data to be taken in Hz. The
            allowed values are 10 to 10e6 Hz for the Moku:Pro for 1
            channel logging, where the max decreases to 5e6 Hz for 2
            channel logging and to 1.25e6 for 3 or 4 channel logging.
            For the Moku:Lab, the allowed values are 10 to 2.5e5 Hz.
            For the Moku:Go, the allowed values are 10 to 1e6 Hz for
            1 channel logging, where the maximum allowed value
            decreases to 500e3 Hz for 2 channel logging. Default is the
            lowest allowed maximum value for the given device (i.e.
            1.25e6 for Moku:Pro, 2.5e5 for Moku:Lab, and 500e3 for
            Moku:Go).
        max_duration_per_file : float, optional
            The maximum amount of seconds to save to a single file.
            Avoids creating very large single files. Default is 60
            seconds.
        acquisition_mode : str, optional
            Changes acquisition mode between 'Normal' and 'Precision'.
            Precision mode is also known as decimation, it samples at
            the full rate and averages excess data points to improve
            precision. Normal mode works by direct down sampling,
            throwing away extra data points. Default is 'Precision'.

        """

        self.DL = Datalogger(
            ip_address, force_connect=force_connect, session_trust_env=False,
        )
        self.DL.set_acquisition_mode(acquisition_mode)
        self._max_dur_per_file = max_duration_per_file

        self._device = self.DL.describe()['hardware']
        if self._device not in ['Moku:Pro', 'Moku:Lab', 'Moku:Go']:
            raise ValueError(
                "Unrecognized device, is not a Moku:Go, Moku:Lab, or Moku:Pro."
            )

        if fs is None:
            if self._device == "Moku:Pro":
                self._fs = 1.25e6
            elif self._device == "Moku:Lab":
                self._fs = 2.5e5
            elif self._device == "Moku:Go":
                self._fs = 500e3
        else:
            self._fs = fs

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
        elif self._device in ["Moku:Lab", "Moku:Go"]:
            chan_list = [1, 2]

        for chan in chan_list:
            self.DL.enable_input(chan, False) # disable inputs
            self.DL.generate_waveform(chan, 'Off') # disable outputs

        self.DL.relinquish_ownership()

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
            Moku:Lab and Moku:Go, valid channel numbers are 1 and 2.
        impedance : str, list of str, optional
            The output impedance to use for each channel. For the
            Moku:Pro and Moku:Lab, options are '1MOhm' and '50Ohm'. For
            the Moku:Go, the only option is '1MOhm'. Can pass a list of
            the same length as channels if different impedances are
            desired for a Moku:Pro. Default is '1MOhm' for all channels.
        coupling : str, list of str, optional
            The coupling to use for each channel. Options are 'DC' and
            'AC'. Can pass a list of the same length as channels if
            different couplings are desired. Default is 'DC' for all
            channels.
        vrange : str, list of str, optional
            The voltage range to use for each channel. For the
            Moku:Pro, options are '400mVpp', '4Vpp' and '40Vpp'. For
            the Moku:Lab, the options are '1Vpp' and '10Vpp'. For the
            Moku:Go, options are '10Vpp' and '50Vpp'. Can pass a list
            of the same length as channels if different voltage ranges
            are desired for specific channels. Default is the smallest
            allowed value for the device for all channels (i.e.
            '400mVpp' for the Moku:Pro, '1Vpp' for the Moku:Lab, and
            '10Vpp' for the Moku:Go).

        """

        if self._device == "Moku:Pro":
            chan_list = [1, 2, 3, 4]
            vrange = '400mVpp' if vrange is None else vrange
        elif self._device == "Moku:Lab":
            chan_list = [1, 2]
            vrange = '1Vpp' if vrange is None else vrange
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
                self.DL.enable_input(chan, False)
            else:
                ind = channels.index(chan)
                self.DL.set_frontend(
                    channel=chan,
                    impedance=impedance[ind],
                    coupling=coupling[ind],
                    range=vrange[ind],
                )

    @staticmethod
    def sine_settings(amplitude=1, frequency=10000, offset=0, phase=0,
                      symmetry=50):
        """
        Method to return a dictionary with the valid sine wave
        settings to be passed to `self.set_output_channel`.

        Parameters
        ----------
        amplitude : float, optional
            Waveform peak-to-peak amplitude in Volts. Allowed values
            are 2e-3 to 10 for the Moku:Go, 2e-3 to 4 for the
            Moku:Lab, and 1e-3 to 10 for the Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 1 V.
        frequency : float, optional
            Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
            for the Moku:Go, 1e-3 to 100e6 for the Moku:Lab, and 1e-3
            to 500e6 for the Moku:Pro. Default is 10000 (1e4) Hz.
        offset : float, optional
            DC offset applied to the waveform in V. Allowed values are
            -5 to 5 for Moku:Go, Moku:Lab, and Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go, Moku:Lab, and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go, Moku:Lab, and Moku:Pro. Default is
            50.

        Returns
        -------
        settings_dict : dict

        """

        return {
            'amplitude': amplitude,
            'frequency': frequency,
            'offset': offset,
            'phase': phase,
            'symmetry': symmetry,
        }

    @staticmethod
    def square_settings(amplitude=1, frequency=10000, offset=0, phase=0,
                        duty=50, symmetry=50):
        """
        Method to return a dictionary with the valid square wave
        settings to be passed to `self.set_output_channel`.

        Parameters
        ----------
        amplitude : float, optional
            Waveform peak-to-peak amplitude in Volts. Allowed values
            are 2e-3 to 10 for the Moku:Go, 2e-3 to 4 for the
            Moku:Lab, and 1e-3 to 10 for the Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 1 V.
        frequency : float, optional
            Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
            for the Moku:Go, 1e-3 to 100e6 for the Moku:Lab, and 1e-3
            to 500e6 for the Moku:Pro. Default is 10000 (1e4) Hz.
        offset : float, optional
            DC offset applied to the waveform in V. Allowed values are
            -5 to 5 for Moku:Go, Moku:Lab, and Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go, Moku:Lab, and Moku:Pro. Default is 0.
        duty : float, optional
            Duty cycle as percentage in %. Allowed values are 0 to 100
            for Moku:Go and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go, Moku:Lab, and Moku:Pro. Default is
            50.

        Returns
        -------
        settings_dict : dict
            A dictionary containing all of the required settings
            needed to set up a square wave output.

        """

        return {
            'amplitude': amplitude,
            'frequency': frequency,
            'offset': offset,
            'phase': phase,
            'duty': duty,
            'symmetry': symmetry,
        }

    @staticmethod
    def ramp_settings(amplitude=1, frequency=10000, offset=0, phase=0,
                      symmetry=50):
        """
        Method to return a dictionary with the valid ramp wave
        settings to be passed to `self.set_output_channel`.

        Parameters
        ----------
        amplitude : float, optional
            Waveform peak-to-peak amplitude in Volts. Allowed values
            are 2e-3 to 10 for the Moku:Go, 2e-3 to 4 for the
            Moku:Lab, and 1e-3 to 10 for the Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 1 V.
        frequency : float, optional
            Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
            for the Moku:Go, 1e-3 to 100e6 for the Moku:Lab, and 1e-3
            to 500e6 for the Moku:Pro. Default is 10000 (1e4) Hz.
        offset : float, optional
            DC offset applied to the waveform in V. Allowed values are
            -5 to 5 for Moku:Go, Moku:Lab, and Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go, Moku:Lab, and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go, Moku:Lab, and Moku:Pro. Default is
            50.

        Returns
        -------
        settings_dict : dict
            A dictionary containing all of the required settings
            needed to set up a ramp wave output.

        """

        return {
            'amplitude': amplitude,
            'frequency': frequency,
            'offset': offset,
            'phase': phase,
            'symmetry': symmetry,
        }

    @staticmethod
    def pulse_settings(amplitude=1, frequency=10000, offset=0, phase=0,
                       symmetry=50, edge_time=0, pulse_width=0):
        """
        Method to return a dictionary with the valid pulse train
        settings to be passed to `self.set_output_channel`.

        Parameters
        ----------
        amplitude : float, optional
            Waveform peak-to-peak amplitude in Volts. Allowed values
            are 2e-3 to 10 for the Moku:Go, 2e-3 to 4 for the
            Moku:Lab, and 1e-3 to 10 for the Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 1 V.
        frequency : float, optional
            Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
            for the Moku:Go, 1e-3 to 100e6 for the Moku:Lab, and 1e-3
            to 500e6 for the Moku:Pro. Default is 10000 (1e4) Hz.
        offset : float, optional
            DC offset applied to the waveform in V. Allowed values are
            -5 to 5 for Moku:Go, Moku:Lab, and Moku:Pro (For Moku:Pro,
            the output voltage is limited to between -1V and 1V above
            1MHz). Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go, Moku:Lab, and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go, Moku:Lab, and Moku:Pro. Default is
            50.
        edge_time : float, optional
            Edge-time of the waveform in s. Allowed values are 16e-9 to
            pulse_width for the Moku:Go, 4e-9 to pulse_width for the
            Moku:Lab, and 2e-9 to pulse_width for the Moku:Pro. Default
            is 0 (which sets it to the shortest allowed time).
        pulse_width : float, optional
            Pulse width of the waveform in s. Allowed values are 16e-9
            to waveform period for the Moku:Go, 4e-9 to waveform period
            for the Moku:Lab and 2e-9 to waveform period for the
            Moku:Pro. Default is 0 (which sets it to the shortest
            allowed time).

        Returns
        -------
        settings_dict : dict
            A dictionary containing all of the required settings
            needed to set up a pulse train output.

        """

        return {
            'amplitude': amplitude,
            'frequency': frequency,
            'offset': offset,
            'phase': phase,
            'symmetry': symmetry,
            'edge_time': edge_time,
            'pulse_width': pulse_width,
        }

    @staticmethod
    def dc_settings(dc_level=0):
        """
        Method to return a dictionary with the valid DC value
        settings to be passed to `self.set_output_channel`.

        Parameters
        ----------
        dc_level : float, optional
            The DC level of the output of the channel in V.
            Default is 0.

        Returns
        -------
        settings_dict : dict
            A dictionary containing all of the required settings
            needed to set up a DC output.

        """

        return {
            'dc_level': dc_level,
        }

    def set_output_channel(self, channel, waveformtype, load="HiZ",
                           **settings):
        """
        Method to turn on an output channel and generate the speicified
        waveform. Use the various self.*_settings() methods to
        generate the various valid settings. Can only set one channel
        at a time.

        Parameters
        ----------
        channel : int
            The output channel to generate a waveform on. With a
            Moku:Pro, this must be an integer of 1, 2, 3 or 4. With a
            Moku:Go or Moku:Lab, this must be an integer of 1 or 2.
        waveformtype : str
            The waveform type to generate, must be one of 'Sine',
            'Square', 'Ramp', 'Pulse', 'DC'. Can also be set to 'Off'
            to turn the channel off.
        termination : str, optional
            The waveform termination to use for the output channel.
            For a Moku:Pro or Moku:Lab, this must be one of 'HiZ' or
            '50Ohm'. For a Moku:Go, this can only be 'HiZ'. Default
            is 'HiZ'.
        settings : dict
            The dictionary containing all of the settings needed for
            the specified waveform type.

        """

        if self._device in ["Moku:Pro", "Moku:Lab"]:
            self.DL.set_output_termination(channel, load)
        self.DL.generate_waveform(
            channel,
            waveformtype,
            **settings,
        )


    def log_data(self, duration, savepath='./', comments='',
                 file_name_prefix=''):
        """
        Method to log data after setting up the desired configuration.
        Saves a LI file to the specified save path.

        Parameters
        ----------
        duration : integer
            The total number of seconds to log data with the current
            configuration. Must be an integer.
        savepath : str, optional
            The absolute path on to save the LI file after it has been
            logged by the Moku. Default is the current working
            directory.
        comments : str, optional
            Any comments that should be added to the metadata of the
            file to be saved. Default is an empty string.
        file_name_prefix : str, optional
            Prefix to use in the filename. Default is set by the Moku
            as "MokuDataLoggerData".

        """

        self.DL.set_samplerate(self._fs)

        filenames = []
        nfiles = np.int32(np.ceil(duration / self._max_dur_per_file))

        for ii in range(nfiles):
            if (ii + 1) * self._max_dur_per_file > duration:
                this_file_duration = duration % self._max_dur_per_file
            else:
                this_file_duration = self._max_dur_per_file

            logfile = self.DL.start_logging(
                duration=this_file_duration,
                comments=comments,
                file_name_prefix=file_name_prefix,
            )
            # Track progress percentage of the data logging session
            is_logging = True
            while is_logging:
                # Wait for the logging session to progress by sleeping 0.5sec
                time.sleep(0.5)
                # Get current progress percentage and check if it's complete
                progress = self.DL.logging_progress()
                is_logging = not progress['complete']

            filenames.append(logfile['file_name'])

        if self._device == "Moku:Pro":
            target = "ssd"
        elif self._device == "Moku:Lab":
            target = "media"
        elif self._device == "Moku:Go":
            target = "persist"

        for fname in filenames:
            self.DL.download(
                target,
                fname,
                os.path.abspath(savepath + os.sep + fname),
            )
