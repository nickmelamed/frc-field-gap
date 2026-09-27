import argparse
import logging

import pytest

from frc_xdata.logging_utils import add_log_level_argument, setup_logging


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_log_level_argument(parser)
    return parser


def test_log_level_defaults_to_info() -> None:
    assert make_parser().parse_args([]).log_level == "INFO"


def test_log_level_accepts_lowercase() -> None:
    assert make_parser().parse_args(["--log-level", "debug"]).log_level == "DEBUG"


def test_log_level_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        make_parser().parse_args(["--log-level", "loud"])


def test_setup_logging_sets_root_level() -> None:
    setup_logging("warning")
    assert logging.getLogger().level == logging.WARNING
