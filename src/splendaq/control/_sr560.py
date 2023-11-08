import numpy as np
import serial


__all__ = [
    "SR560",
]


class SR560(object):
    """
    Class for controlling an SRS SR560 low-noise preamplifier.
    Documentation on the various commands can be found in the SR560
    user manual.
    
    """
    
    def __init__(self, port):
        """
        Initialization of the SS560 class. Establishes a connection to
        a specified port.

        Parameters
        ----------
        port : str
            The port to connect to that the SR560 is on,
            e.g. "COM4".

        Returns
        -------
        None

        """

        self._SRS = serial.Serial()
        self._SRS.port = port
        self._SRS.open()

    def set_coupling(self, value):
        """
        Set the coupling of the SR560.

        Parameters
        ----------
        value : int
            Sets the input coupling. Coupling to ground is 0,
            DC coupling is 1, and AC coupling is 2.

        Returns
        -------
        None

        """

        self._SRS.write(f"CPLG{value}\r\n".encode())

    def set_dynamic_reserve(self, value):
        """
        Set the dynamic reserve of the SR560.

        Parameters
        ----------
        value : int
            Sets the dynamic reserve (DR). Low noise is 0,
            high DR is 1, and calibration gains is 2.

        Returns
        -------
        None

        """

        self._SRS.write(f"DYNR{value}\r\n".encode())

    def set_blanking(self, value):
        """
        Operates amplifier blanking.

        Parameters
        ----------
        value : int
            Set 0 to for not blanked. Set to 1 for blanked.

        Returns
        -------
        None

        """

        self._SRS.write(f"BLINK{value}\r\n".encode())

    def set_filter_mode(self, value):
        """
        Sets filter mode.

        Parameters
        ----------
        value : int
            The filter mode to set, values below
            0 = bypass
            1 =  6 dB lowpass
            2 = 12 dB lowpass
            3 =  6 dB highpass
            4 = 12 dB highpass
            5 = bandpass

        Returns
        -------
        None

        """

        self._SRS.write(f"FLTM{value}\r\n".encode())
        
    def set_gain(self, gain):
        """
        Controls the gain of the SR560.
        
        Parameters
        ----------
        gain : int
            Sets the gain of the device, allowed values are
            1 to 50000 in 1-2-5 sequence.

        Returns
        -------
        None

        """

        gain_dict = {
            1 : 0,
            2 : 1,
            5 : 2,
            10 : 3,
            20 : 4,
            50 : 5,
            100 : 6,
            200 : 7,
            500 : 8,
            1000 : 9,
            2000 : 10,
            5000 : 11,
            10000 : 12,
            20000 : 13,
            50000 : 14,
        }

        if gain not in gain_dict:
            raise ValueError(
                "Not a valid gain, see docstring for allowed values."
            )

        value = gain_dict[gain]
        self._SRS.write(f"GAIN{value}\r\n".encode())

    def set_highpass_filter(self, freq):
        """
        Set the highpass filter frequency.

        Parameters
        ----------
        freq : float
            The highpass filter frequency to be set. Defined in steps
            from 0.03, 0.1, 0.3, 1, ..., 10000 Hz.

        Returns
        -------
        None

        """

        freq_dict = {
            0.03 : 0,
            0.1 : 1,
            0.3 : 2,
            1 : 3,
            3 : 4,
            10 : 5,
            30 : 6,
            100 : 7,
            300 : 8,
            1000 : 9,
            3000 : 10,
            10000 : 11,
        }

        if freq not in freq_dict:
            raise ValueError(
                "Not a valid highpass freq, "
                "see docstring for allowed values."
            )

        value = freq_dict[freq]
        self._SRS.write(f"HFRQ{value}\r\n".encode())

    def set_signal_invert(self, value):
        """
        Sets the signal invert status.

        Parameters
        ----------
        value : int
            0 is noninverted. 1 is inverted.

        Returns
        -------
        None

        """

        self._SRS.write(f"INVT{value}\r\n".encode())

    def listen_all(self):
        """
        Makes all attached SR560s listeners.

        Returns
        -------
        None

        """

        self._SRS.write(b"LALL\r\n")

    def listen(self, value):
        """
        Listen to a specific SR560.

        Parameters
        ----------
        value : int
            Makes SR560 with address `value` a listener.
            Can be 0, 1, 2, or 3.

        Returns
        -------
        None

        """

        self._SRS.write(f"LISN{value}\r\n".encode())

    def set_lowpass_filter(self, freq):
        """
        Set the lowpass filter frequency.

        Parameters
        ----------
        freq : float
            The lowpass filter frequency to be set. Defined in steps
            from 0.03, 0.1, 0.3, 1, ..., 1000000 Hz.

        Returns
        -------
        None

        """

        freq_dict = {
            0.03 : 0,
            0.1 : 1,
            0.3 : 2,
            1 : 3,
            3 : 4,
            10 : 5,
            30 : 6,
            100 : 7,
            300 : 8,
            1000 : 9,
            3000 : 10,
            10000 : 11,
            30000 : 12,
            100000 : 13,
            300000 : 14,
            1000000 : 15,
        }

        if freq not in freq_dict:
            raise ValueError(
                "Not a valid lowpass freq, "
                "see docstring for allowed values."
            )

        value = freq_dict[freq]
        self._SRS.write(f"LFRQ{freq}\r\n".encode())

    def reset_overload(self):
        """
        Resets overload for 0.5 seconds.

        Returns
        -------
        None

        """

        self._SRS.write(b"ROLD\r\n")

    def set_input_source(self, value):
        """
        Sets the input source.

        Parameters
        ----------
        value : int
            Input source. Set 0 for "A",
            set to 1 for "A-B", and set to 2 for "B".

        Returns
        -------
        None

        """

        self._SRS.write(f"SRCE{value}\r\n".encode())

    def set_vernier_gain_status(self, value):
        """
        Set the vernier gain status.

        Parameters
        ----------
        value : int
            Set to 0 for cal'd gain. Set to 1 for vernier gain.

        Returns
        -------
        None

        """

        self._SRS.write(f"UCAL{value}\r\n".encode())

    def set_vernier_gain(self, gain):
        """
        Set the vernier gain on the SR560.

        Parameters
        ----------
        gain : float
            Set the vernier gain in 0.5% steps from 0 to 100.

        Returns
        -------
        None

        """
        
        allowed_vals = np.linspace(0, 100, num=201)

        if gain not in allowed_vals:
            raise ValueError(
                "Vernier gain must be between 0 and 100 in steps of 0.5."
            )

        self._SRS.write(f"UCGN{gain}\r\n".encode())

    def unlisten_all(self):
        """
        Unlisten. Unaddresses all attached SR560s.

        Returns
        -------
        None

        """

        self._SRS.write(b"UNLS\r\n")

    def reset_to_defaults(self):
        """
        Reset. Recalls default settings.

        Returns
        -------
        None

        """

        self._SRS.write(b"*RST\r\n")

    def close(self):
        """
        Close connection to the port to free it for other users.

        Returns
        -------
        None

        """
        
        self._SRS.close()
