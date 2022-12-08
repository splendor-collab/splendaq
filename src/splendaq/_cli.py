import argparse
import os
import sys
import yaml

import splendaq
from splendaq.daq import Sequencer

def splendaq_cli():
    """
    SPLENDAQ command line interface.

    """

    parser = argparse.ArgumentParser(
        prog="splendaq", description="splendaq's command-line interface"
    )

    # global options
    parser.add_argument(
        "--version", action="store_true", help="Print splendaq version and exit",
    )

    subparsers = parser.add_subparsers()

    add_sequencer_parser(subparsers)

    args = parser.parse_args()

    if args.version:
        print(splendaq.__version__)
        sys.exit()

    args.func(args)


def add_sequencer_parser(subparsers):
    """
    Configure the DC sequencer command line interface.

    """

    parser_sequencer = subparsers.add_parser(
        "sequencer", description="Run a DC sequencer on the Moku, logging input channels",
    )

    parser_sequencer.add_argument(
        "yaml_file", help="YAML file containing sequencer setup.",
    )

    parser_sequencer.add_argument(
        "--noprint",
        action="store_false",
        help="""Do not print output DC voltage values.""",
    )

    parser_sequencer.set_defaults(func=sequencer_cli)

def sequencer_cli(args):
    """
    Sequencer command line interface.
    
    """

    SEQ = Sequencer(args.yaml_file)
    SEQ.run(args.noprint)
