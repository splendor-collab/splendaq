from datetime import datetime
import itertools
import numpy as np
import os
import time
import yaml

from splendaq.daq import LogData, Oscilloscope


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
        self._savepath = self.yaml_dict['moku']['savepath']

    def run(self, verbose=True):
        """
        Start running the sequencer.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints out information on the current point
            being run by the sequencer.

        """

        if self.yaml_dict['moku']['sequencer_mode']=='full':
            self._full_run(verbose)
        elif self.yaml_dict['moku']['sequencer_mode']=='quick':
            self._quick_run(verbose)
        else:
            raise ValueError("Unrecognized sequencer mode type.")

    def _init_run(self):
        """Initialize inputs and outputs based on the YAML file."""

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

        return input_list, input_settings, output_list, output_ranges

    @staticmethod
    def _print_status(outputs, output_list):
        """Hidden method to print current bias points of outputs."""
        print(
            ', '.join(
                [
                    f"Output{b} = {a} V" for a, b in zip(
                        outputs,
                        output_list,
                    )
                ]
            )
        )

    def _full_run(self, verbose):
        """
        Running the 'full' sequencer mode, which calls the LogData
        class to save the time-domain traces to LI files.

        """

        input_list, input_settings, output_list, output_ranges = self._init_run()

        with LogData(
                self.yaml_dict['moku']['ip'],
                force_connect=self.yaml_dict['moku']['force_connect'],
                acquisition_mode=self.yaml_dict['moku']['acquisition_mode'],
            ) as LOG:

            LOG.set_input_channels(input_list, **input_settings)

            for outputs in itertools.product(*output_ranges):

                for ii in output_list:
                    dc_settings = LOG.dc_settings(
                        dc_level=outputs[output_list.index(ii)],
                    )
                    LOG.set_output_channel(ii, 'DC', **dc_settings)

                time.sleep(self.yaml_dict['moku']['time_between_points'])

                if verbose:
                    self._print_status(outputs, output_list)

                LOG.log_data(
                    self.yaml_dict['moku']['duration_per_point'],
                    savepath=self._savepath,
                    file_name_prefix="splendaq_iv",
                )

    def _quick_run(self, verbose):
        """
        Running the 'quick' sequencer mode, which calls the
        Oscilloscope class to simply save DC values at each bias point.

        """

        input_list, input_settings, output_list, output_ranges = self._init_run()
        product_len = np.multiply(*[len(arr) for arr in output_ranges])
        data_array = np.zeros(
            (product_len, len(input_list) + len(output_list))
        )
        columns = [
            f"Input{ii}" for ii in input_list
        ] + [
            f"Output{ii}" for ii in input_list
        ]

        with Oscilloscope(
                self.yaml_dict['moku']['ip'],
                force_connect=self.yaml_dict['moku']['force_connect'],
                acquisition_mode=self.yaml_dict['moku']['acquisition_mode'],
            ) as Osc:

            Osc.set_input_channels(input_list, **input_settings)

            for n, outputs in enumerate(itertools.product(*output_ranges)):

                Osc.set_output_channels(output_list, 'DC', dc_level=outputs)

                time.sleep(self.yaml_dict['moku']['time_between_points'])

                if verbose:
                    self._print_status(outputs, output_list)

                data = Osc.get_data(
                    [f"Input{ii}" for ii in input_list],
                    self.yaml_dict['moku']['duration_per_point'],
                )

                data_array[n, :len(input_list)] = [
                    np.mean(data[f'ch{ii + 1}']) for ii in range(len(input_list))
                ]
                data_array[n, len(input_list):] = outputs

        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        np.savetxt(
            f'{self._savepath}{os.sep}splendaq_iv_{date_str}.txt',
            data_array,
            delimiter=',',
            header=','.join(columns),
        )

