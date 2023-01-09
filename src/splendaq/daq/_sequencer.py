import itertools
import numpy as np
import yaml

from splendaq.daq import LogData

__all__ = [
    "Sequencer",
]


class Sequencer(object):
    """
    Class for running a DC sequencer with various inputs read out,
    while changing various outputs by specified steps.

    Attributes
    ----------
    yaml_dict : dict
        A dictionary containing all of the settings to be used for
        taking data with the sequencer.

    """

    def __init__(self, yaml_file):
        """
        Initialization of the Sequencer given a YAML file which
        contains all of the settings to be used.

        Parameters
        ----------
        yaml_file : str
            The absolute path to the YAML file to be used to set up
            the sequencer for data taking.

        """

        with open(yaml_file, 'r') as f:
            self.yaml_dict = yaml.safe_load(f)

    def run(self, verbose=True):
        """
        Begin running the sequencer.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints out information on the current point
            being run by the sequencer.

        """

        input_list = [
            int(k[5:]) for k in self.yaml_dict if (
                "input" in k and self.yaml_dict[k]['log']
            )
        ]

        input_dict = [
            d for k, d in self.yaml_dict.items() if (
                "input" in k and self.yaml_dict[k]['log']
            )
        ]

        input_settings = {
            "vrange": [d['vrange'] for d in input_dict],
            "impedance": [d['impedance'] for d in input_dict],
        }

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
            with LogData(
                self.yaml_dict['moku']['ip'],
                force_connect=self.yaml_dict['moku']['force_connect'],
                acquisition_mode=self.yaml_dict['moku']['acquisition_mode'],
            ) as LOG:
                LOG.set_input_channels(input_list, **input_settings)

                for ii in output_list:
                    dc_settings = LOG.dc_settings(
                        dc_level=outputs[output_list.index(ii)],
                    )
                    LOG.set_output_channel(ii, 'DC', **dc_settings)
                if verbose:
                    print(
                        ''.join(
                            [
                                f"Output{b} = {a} V, " for a, b in zip(
                                    outputs,
                                    output_list,
                                )
                            ]
                        )[:-1]
                    )

                LOG.log_data(
                    self.yaml_dict['moku']['duration_per_file'],
                    file_name_prefix=f"splendaq_iv",
                )
