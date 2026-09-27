import argparse
import logging
from collections.abc import Iterator

import pytest

from frc_xdata.logging_utils import add_log_level_argument, setup_logging


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    # setup_logging replaces the root handlers, which would otherwise leak
    # into every test that runs after this one.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


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


@pytest.mark.usefixtures("restore_root_logger")
def test_setup_logging_sets_root_level() -> None:
    setup_logging("warning")
    assert logging.getLogger().level == logging.WARNING
