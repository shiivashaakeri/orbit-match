# orbit-match/orbitmatch/utils/logging_setup.py
# Run: imported by other modules; not a runnable script.

"""Project-wide logging setup.

Every module obtains its logger via :func:`get_logger`. The first call
configures the root project logger with a consistent format, level, and
handlers; subsequent calls return child loggers that inherit the configuration.

Usage
-----
At the top of any module::

    from orbitmatch.utils.logging_setup import get_logger
    log = get_logger(__name__)

    log.info("Computing feasibility tensor")

In a runnable script, optionally bump the level and add a file handler::

    from orbitmatch.utils.logging_setup import configure, get_logger
    configure(level="DEBUG", log_file="logs/lambda2_traces.log")
    log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_ROOT_LOGGER_NAME = "orbitmatch"
_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure(
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    overwrite: bool = False,
) -> logging.Logger:
    """Configure the root project logger.

    Parameters
    ----------
    level
        Logging level for the project root logger. Either a string
        ("DEBUG", "INFO", "WARNING", "ERROR") or a logging constant.
    log_file
        Optional path to a log file. If given, log records are written there
        in addition to stderr.
    overwrite
        If ``True``, re-configure even if already configured.

    Returns
    -------
    logging.Logger
        The configured project root logger.
    """
    global _CONFIGURED  # noqa: PLW0603

    root = logging.getLogger(_ROOT_LOGGER_NAME)

    if _CONFIGURED and not overwrite:
        return root

    # Remove any pre-existing handlers (e.g. from a re-configure call).
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)
    root.propagate = False  # Don't double-log to the python root logger.

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the project root.

    Auto-configures the root logger on first call with default settings
    (INFO level, stderr-only). Pass ``name=__name__`` from modules so
    log records carry the dotted module path.
    """
    if not _CONFIGURED:
        configure()

    # Strip the leading "orbitmatch." prefix if present, to keep names compact.
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        short = name[len(_ROOT_LOGGER_NAME) + 1 :]
        return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")
    if name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
