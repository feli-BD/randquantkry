##########################################################################
#
#                       Logging Configuration
#
##########################################################################
"""Central logging setup for randquantkry.

Provides dual-channel logging so benchmark runs are fully traceable:

    * a human-readable stream to the console (level configurable), and
    * a machine-parseable JSON-lines file (one JSON object per line) that
      captures every log record with full structured metadata.

Every module in the package should obtain its logger via ``get_logger`` so
that all output flows through the same handlers and a single benchmark run
produces one coherent, timestamped audit trail.

Typical use at the start of a benchmark session::

    from randquantkry.logging_config import configure_logging, get_logger

    configure_logging(json_path="runs/run_2026-06-07.jsonl", console_level="INFO")
    log = get_logger(__name__)
    log.info("session start", extra={"context": {"seed": 42}})
"""

# Imports
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PACKAGE_LOGGER_NAME = "randquantkry"

# A sentinel set of attributes that live on every LogRecord 
_STANDARD_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLinesFormatter(logging.Formatter):
    """Format a LogRecord as a single JSON object (one line).

    Standard fields (timestamp, level, logger, message, source location) are
    always emitted. Any custom keyword passed through ``extra=`` on the logging
    call is merged in, so structured benchmark metadata (matrix dimensions,
    FLOP estimates, timings, seeds, ...) is preserved for later parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Merge any user-supplied structured context.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class ConsoleContextFormatter(logging.Formatter):
    """Human-readable formatter that appends any extra= context."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
        }
        if extras:
            base += f" | {extras}"
        return base


def configure_logging(
    json_path: str | Path | None = None,
    console_level: str | int = "INFO",
    file_level: str | int = "DEBUG",
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure the package logger with console + optional JSON-lines file.

    Parameters
    ----------
    json_path : str or Path, optional
        Destination for the JSON-lines log. Parent directories are created if
        needed. If ``None``, no file handler is attached (console only).
    console_level : str or int, default "INFO"
        Threshold for the human-readable console handler.
    file_level : str or int, default "DEBUG"
        Threshold for the JSON file handler. Kept at DEBUG by default so the
        on-disk audit trail is maximally complete even when the console is quiet.
    capture_warnings : bool, default True
        Route ``warnings.warn`` messages through logging as well.

    Returns
    -------
    logging.Logger
        The configured package-root logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    # The logger itself must pass everything through; handlers do the filtering.
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if configure_logging is called more than once.
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(
        ConsoleContextFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console)

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(json_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(file_handler)
        logger.info("logging configured", extra={"context": {"json_path": str(json_path)}})
    else:
        logger.info("logging configured (console only)")

    logging.captureWarnings(capture_warnings)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the package namespace.

    Parameters
    ----------
    name : str, optional
        Usually ``__name__`` from the calling module. If it already starts with
        the package name it is used as-is; otherwise it is nested under the
        package logger so all records share the configured handlers.
    """
    if not name or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(PACKAGE_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")
