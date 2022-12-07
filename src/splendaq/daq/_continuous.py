import time
import os
import itertools
import numpy as np
import yaml

import moku
from moku import instruments

# monkey patch for fixing proxies
class NewRequestSession(moku.session.RequestSession):
    def __init__(self, ip, force_connect, ignore_busy, persist_state,
                 connect_timeout, read_timeout):
        super().__init__(ip, force_connect, ignore_busy, persist_state,
                         connect_timeout, read_timeout)
        self.rs.proxies.update(
            {
                "https": "",
                "http": "",
            }
        )
        self.rs.trust_env = False

moku.session.RequestSession = NewRequestSession
moku.instruments._datalogger.Moku = moku.Moku
Datalogger = instruments.Datalogger


__all__ = [
    "LogData",
    "Sequencer",
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

    def __init__(self, ip_address, force_connect=False, fs=1.25e6,
                 max_duration_per_file=60, acquisition_mode="Normal"):
        """
        Initialization of the LogData class.

        Parameters
        ----------
        ip_address : str
            The IP address of the Moku to connect to. If connecting via
            USB-C, ensure to enclose the address with square brackets,
            e.g. "[your_ip_address]".
        force_connect : bool, optional
            Take ownership of the Moku even if it is being used by
            someone else. Default is False.
        fs : float, optional
            The digitization rate of the data to be taken in Hz.
            Default is 1.25e6.
        max_duration_per_file : float, optional
            The maximum amount of seconds to save to a single file.
            Avoids creating very large single files. Default is 60
            seconds.
        acquisition_mode : str, optional
            Changes acquisition mode between 'Normal' and 'Precision'.
            Precision mode is also known as decimation, it samples at
            the full rate and averages excess data points to improve
            precision. Normal mode works by direct down sampling,
            throwing away extra data points. Default is 'Normal'.

        """

        self.DL = Datalogger(ip_address, force_connect=force_connect)
        self.DL.set_samplerate(fs)
        self.DL.set_acquisition_mode(acquisition_mode)
        self._max_dur_per_file = max_duration_per_file

    def __enter__(self):
        """Returns self when entered via with."""
        return self

    def __exit__(self, type, value, traceback):
        """Always run the relinquish ownership when exiting."""
        self.DL.relinquish_ownership()

    def set_input_channels(self, channels, impedance="1MOhm", coupling="DC",
                           vrange="400mVpp"):
        """
        Set which channels to save data from and which settings to use.

        Parameters
        ----------
        channels : int, list of int
            Which input channels to log data from. Can be a single
            value for one channel, or a list of the channels. Valid
            channel numbers of 1, 2, 3, and 4.
        impedance : str, list of str, optional
            The output impedance to use for each channel. Options are
            '1MOhm' and '50Ohm'. Can pass a list of the same length as
            channels if different impedances are desired. Default is
            '1MOhm' for all channels.
        coupling : str, list of str, optional
            The coupling to use for each channel. Options are 'DC' and
            'AC'. Can pass a list of the same length as channels if
            different couplings are desired. Default is 'DC' for all
            channels.
        vrange : str, list of str, optional
            The voltage range to use for each channel. Options are
            '400mVpp', '4Vpp' and '40Vpp'. Can pass of list of the same
            length as channels if different voltage ranges are desired
            for specific channels. Default is '400mVpp' for all
            channels.

        """

        chan_list = [1, 2, 3, 4]

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
                self.DL.disable_channel(chan)
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
             are 2e-3 to 10 for the Moku:Go and 1e-3 to 10 for the
             Moku:Pro (For Moku:Pro, the output voltage is limited to
             between -1V and 1V above 1MHz). Default is 1 V.
        frequency : float, optional
             Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
             for the Moku:Go and 1e-3 to 500e6 for the Moku:Pro.
             Default is 10000 Hz.
        offset : float, optional
             DC offset applied to the waveform in V. Allowed values are
             -5 to 5 for Moku:Go and Moku:Pro (For Moku:Pro, the output
             voltage is limited to between -1V and 1V above 1MHz).
             Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go and Moku:Pro. Default is 50.

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
             are 2e-3 to 10 for the Moku:Go and 1e-3 to 10 for the
             Moku:Pro (For Moku:Pro, the output voltage is limited to
             between -1V and 1V above 1MHz). Default is 1 V.
        frequency : float, optional
             Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
             for the Moku:Go and 1e-3 to 500e6 for the Moku:Pro.
             Default is 10000 Hz.
        offset : float, optional
             DC offset applied to the waveform in V. Allowed values are
             -5 to 5 for Moku:Go and Moku:Pro (For Moku:Pro, the output
             voltage is limited to between -1V and 1V above 1MHz).
             Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go and Moku:Pro. Default is 0.
        duty : float, optional
            Duty cycle as percentage in %. Allowed values are 0 to 100
            for Moku:Go and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go and Moku:Pro. Default is 50.

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
             are 2e-3 to 10 for the Moku:Go and 1e-3 to 10 for the
             Moku:Pro (For Moku:Pro, the output voltage is limited to
             between -1V and 1V above 1MHz). Default is 1 V.
        frequency : float, optional
             Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
             for the Moku:Go and 1e-3 to 500e6 for the Moku:Pro.
             Default is 10000 Hz.
        offset : float, optional
             DC offset applied to the waveform in V. Allowed values are
             -5 to 5 for Moku:Go and Moku:Pro (For Moku:Pro, the output
             voltage is limited to between -1V and 1V above 1MHz).
             Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go and Moku:Pro. Default is 50.

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
             are 2e-3 to 10 for the Moku:Go and 1e-3 to 10 for the
             Moku:Pro (For Moku:Pro, the output voltage is limited to
             between -1V and 1V above 1MHz). Default is 1 V.
        frequency : float, optional
             Waveform frequency in Hz. Allowed values are 1e-3 to 20e6
             for the Moku:Go and 1e-3 to 500e6 for the Moku:Pro.
             Default is 10000 Hz.
        offset : float, optional
             DC offset applied to the waveform in V. Allowed values are
             -5 to 5 for Moku:Go and Moku:Pro (For Moku:Pro, the output
             voltage is limited to between -1V and 1V above 1MHz).
             Default is 0.
        phase : float, optional
            Waveform phase offset in degrees. Allowed values are 0 to
            360 for Moku:Go and Moku:Pro. Default is 0.
        symmetry : float, optional
            Fraction of the cycle rising in %. Allowed values are 0.0
            to 100.0 for Moku:Go and Moku:Pro. Default is 50.
        edge_time : float, optional
            Edge-time of the waveform in s. Allowed values are 16e-9 to
            pulse_width for the Moku:Go and 2e-9 to pulse_width for the
            Moku:Pro. Default is 0 (i.e. the shortest time).
        pulse_width : float, optional
            Pulse width of the waveform in s. Allowed values are 16e-9
            to waveform period for the Moku:Go and 2e-9 to waveform
            period for the Moku:Pro. Default is 0 (i.e. ths shortest
            time).

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

    def set_output_channel(self, channel, waveformtype, load="1MOhm",
                           **settings):
        """
        Method to turn on an output channel and generate the speicified
        waveform. Use the various self.*_settings() methods to
        generate the various valid settings. Can only set one channel
        at a time.

        Parameters
        ----------
        channel : int
            The output channel to generate a waveform on. Must be an
            integer of 1, 2,3 or 4.
        waveformtype : str
            The waveform type to generate, must be one of 'Sine',
            'Square', 'Ramp', 'Pulse', 'DC'.
        load : str, optional
            The load impedance to use for the output channel. Must be
            one of '1MOhm' or '50Ohm'. Default is '1MOhm'.
        settings : dict
            The dictionary containing all of the settings needed for
            the specified waveform type

        """

        self.DL.set_output_load(channel, load)
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
        duration : float
            The total number of seconds to log data with the current
            configuration.
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

        filenames = []
        nfiles = np.int32(np.ceil(duration / self._max_dur_per_file))

        for ii in range(nfiles):
            logfile = self.DL.start_logging(
                duration=duration,
                comments=comments,
                file_name_prefix=file_name_prefix,
            )
            # Track progress percentage of the data logging session
            is_logging = True
            while is_logging:
                # Wait for the logging session to progress by sleeping 0.5sec
                time.sleep(0.5)
                # Get current progress percentage and check if it's complete
                try:
                    progress = self.DL.logging_progress()
                    remaining_time = progress['time_remaining']
                    is_logging = remaining_time >= 0
                except:
                    is_logging = False

            filenames.append(logfile['file_name'])

        for fname in filenames:
            self.DL.download(
                "ssd",
                fname,
                os.path.abspath(savepath + os.sep + fname),
            )

class Sequencer(object):

    def __init__(self, yaml_file):
        
        with open(yaml_file, 'r') as f:
            self.yaml_dict = yaml.safe_load(f)

    def run(self):
        input_list = [
            int(k[5:]) for k in self.yaml_dict if (
                "input" in k and self.yaml_dict[k]['log']
            )
        ]

        output_list = [
            int(k[6:]) for k in self.yaml_dict if (
                "output" in k and self.yaml_dict[k]['apply']
            )
        ]
        output_ranges = [
            np.linspace(
                val['vstart'],
                val['vend'],
                num=val['nstep'],
            ) for k, val in self.yaml_dict.items() if (
                "output" in k and self.yaml_dict[k]['apply']
            )
        ]

        for outputs in itertools.product(*output_ranges):
            with LogData(self.yaml_dict['moku']['ip']) as LOG:
                LOG.set_input_channels(input_list)

                for ii in output_list:
                    dc_settings = LOG.dc_settings(
                        dc_level=outputs[output_list.index(ii)],
                    )
                    LOG.set_output_channel(ii, 'DC', **dc_settings)
                print(''.join([f"Output{b} = {a} V," for a, b in zip(outputs, output_list)])[:-1])

                LOG.log_data(
                    self.yaml_dict['moku']['duration_per_file'],
                    file_name_prefix=f"splendaq_iv",
                )