"""Logging setup shared by the command-line entry points."""

import argparse
import logging

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add a ``--log-level`` option to a command-line parser."""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LEVELS,
        type=str.upper,
        help="logging verbosity (default: INFO)",
    )


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with one timestamped format for every module."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
