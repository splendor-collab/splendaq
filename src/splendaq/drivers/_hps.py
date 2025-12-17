import serial

__all__ = [
    "HarvardPowerSupply",
]

class HarvardPowerSupply(object):
    """
    Class for controlling the Harvard Power Supply at the SLAC
    SPLENDOR lab.

    Notes
    -----
    The documentation of the power supply can be found here
    https://wiki.harvard.edu/confluence/display/ESHOP/TT075%3A+Stanford+HEMT+supply

    The serial commands are reprinted below in case the above link no
    longer works, or the use has no access.

    Commands:

    S <arg>;  selects active slot.  Valid arguments are 1 and 2.
        Power-on default is 1.
    C <arg>; selects active HEMT channel.  Valid arguments are 1 and 2.
        Power-on default is 1.
    D <arg>; sets drain voltage of active HEMT channel in millivolts.
        Valid arguments are floats in the range of 0.0 to 4990.0.
    d; reads back drain voltage of selected HEMT channel.  Note that
        this is mostly for diagnostics, as the "D" command is at least
        as accurate as the "d" command.
    r; reads back remote voltage (Vds+ - Vds-) of selected HEMT channel.
        Mostly for diagnostics unless you anticipate significant
        resistance on the Vds line to the HEMT.
    G <arg>; sets the Vgs in millivolts. Valid arguments are in the
        range of -5000.0 mV to +5000.0 mV.   Power-on default sets Vgs
        to 0.0V.
    g; reads back the previously set Gate voltage.  This is a
        firmware-only operation, simply returning the previously set
        Vgs, rather than measuring it.
    N; turns oN the Vds driver of both HEMTs of the active slot.
        Power-on default has the drivers turned off .
    F; turns ofF the Vds driver of both HEMTs of the active slot.
    i; returns the drain current of the selected HEMT, in mA.

    The preferred protocol is to use "D" to set the correct Vds,
    then use "N" to enable the output drivers.

    """

    def __init__(self, port):
        """
        Initialization of the HarvardPowerSupply class. Establishes
        a connection to a specified port.

        Parameters
        ----------
        port : str
            The port to connect to that the HarvardPowerSupply is on,
            e.g. "COM4".

        """

        self._HPS = serial.Serial()
        self._HPS.port = port
        self._HPS.open()
        self.nslots = 2
        self.nhemts = 2

    def choose_slot(self, slot):
        """
        Choose which slot to control.

        Parameters
        ----------
        slot : int
            The slot to connect to, should be 1 or 2.

        Returns
        -------
        slot_val : int
            The value of the slot that was set.

        """

        self._HPS.write(f"S{slot};".encode())
        slot_str = self._HPS.read_until(b"!")
        self._HPS.write(b'N;')
        self._HPS.read_until(b'!')
        return int(slot_str[1:-1])

    def choose_hemt(self, hemt):
        """
        Choose which HEMT to control on the specified slot.

        Parameters
        ----------
        hemt : int
            The HEMT to connect to on the previously specified slot,
            should be 1 or 2.

        Returns
        -------
        hemt_val : int
            The value of the HEMT that was set.

        """

        self._HPS.write(f'C{hemt};'.encode())
        return int(self._HPS.read_until(b'!')[1:-1])

    def set_gate_voltage(self, value):
        """
        With HEMT and slot chosen, set the gate voltage on the HEMT.

        Parameters
        ----------
        value : float
            The gate voltage to be set for the HEMT, in mV. Allowed
            values are -5000.0 to 5000.0, power-on defaults to 0.

        Returns
        -------
        voltage_val : float
            The value of the gate voltage that was set, in mV.

        """

        self._HPS.write(f"G{value};".encode())
        return float(self._HPS.read_until(b'!')[1:-1])

    def read_gate_voltage(self):
        """
        With HEMT and slot chosen, read the gate voltage on the HEMT.

        Returns
        -------
        voltage_val : float
            The value of the gate voltage that was set, in mV.

        """

        self._HPS.write(b"g;")
        return float(self._HPS.read_until(b'!')[1:-1])

    def set_drain_voltage(self, value):
        """
        With HEMT and slot chosen, set the drain voltage on the HEMT.

        Parameters
        ----------
        value : float
            The drain voltage to be set for the HEMT, in mV. Allowed
            values are 0.0 to 4990.0.

        Returns
        -------
        voltage_val : float
            The value of the gate voltage that was set, in mV.

        """

        self._HPS.write(f"D{value};".encode())
        return float(self._HPS.read_until(b'!')[1:-1])
    
    def read_drain_voltage(self):
        """
        With HEMT and slot chosen, read the drain voltage on the HEMT.

        Returns
        -------
        voltage_val : float
            The value of the drain voltage that was set, in mV.

        """

        self._HPS.write(b"d;")
        return float(self._HPS.read_until(b'!')[1:-1])

    def read_remote_voltage(self):
        """
        With HEMT and slot chosen, read the remote voltage
        (Vds+ - Vds-) on the HEMT.

        Returns
        -------
        voltage_val : float
            The value of the remote voltage in mV.

        """

        self._HPS.write(b"d;")
        return float(self._HPS.read_until(b'!')[1:-1])

    def read_drain_current(self):
        """
        With HEMT and slot chosen, read the drain current on the HEMT.

        Returns
        -------
        current_val : float
            The value of the drain current in mA.

        """

        self._HPS.write(b"i;")
        return float(self._HPS.read_until(b'!')[1:-1])
    
    def zero_all(self):
        """
        Function to zero all voltages, good to run at beginning and end
        of running to ensure voltages are zero and the environment is
        known.

        Returns
        -------
        None

        """

        for slot in range(1, self.nslots + 1):
            self.choose_slot(slot)
            for hemt in range(1, self.nhemts + 1):
                self.choose_hemt(hemt)
                self.set_gate_voltage(0)
                self.set_drain_voltage(0)

    def close(self):
        """
        Close connection to the port to free it for other users.

        Returns
        -------
        None

        """

        self._HPS.close()
