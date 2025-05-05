#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import time
from argparse import SUPPRESS, ArgumentParser

from ..utils import dump_config
from .tau_help_formatter import TauHelpFormatter


def build_argparser():
    """Build an argument parser for this script."""
    parser = ArgumentParser(
        description="Dump the config for an LSST pipeline YAML in YAML format.",
        formatter_class=TauHelpFormatter,
        add_help=False,  # We add our own help to place it at the bottom.
        argument_default=SUPPRESS,
    )
    parser.add_argument(
        "pipeline",
        type=str,
        help="Location of a pipeline definition file in YAML format.",
        metavar="PIPELINE",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="Path to save the config YAML to. Prints to stdout if not provided.",
        metavar="FILE",
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite the output YAML file if it already exists.",
        action="store_true",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        help="Add instrument overrides. Must be a fully qualified class name.",
        metavar="instrument",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Config override for a task, in the format 'label:key=value'.",
        action="append",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    return parser


def main():
    """Use this as the main entry point when calling from the command line."""
    # Set up logging.
    tz = time.strftime("%z")
    logging.basicConfig(
        format="%(levelname)s %(asctime)s.%(msecs)03d" + tz + " - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    args = build_argparser().parse_args()
    if hasattr(args, "filename"):
        if os.path.exists(args.filename):
            if not hasattr(args, "overwrite"):
                raise RuntimeError(f"File {args.filename} already exists; use --overwrite to write anyway.")
            else:
                logger.warning("File %s already exists; overwriting.", args.filename)
        config = dump_config(**{k: v for k, v in vars(args).items() if k not in ["filename", "overwrite"]})
        with open(args.filename, "w") as f:
            f.write(config)
        logger.info(
            "Pipeline config saved at %s.",
            os.path.realpath(args.filename),
        )
    else:
        config = dump_config(**{k: v for k, v in vars(args).items() if k not in ["filename", "overwrite"]})
        print("\n", config, sep="")


if __name__ == "__main__":
    parser = build_argparser()
    parser.parse_args()
