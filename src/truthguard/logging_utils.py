""" Functions used for logging """

from __future__ import annotations

import logging

logger = logging.getLogger("truthguard")
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

_configured = False


def enable_logging(log_file: str = "truthguard.log", level: str = "INFO", also_console: bool = False) -> None:
    global _configured
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if also_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _configured = True
    logger.info("=" * 70)
    logger.info(f"truthguard logging started (level={level}, file={log_file!r})")
    logger.info("=" * 70)
    logger.info("")


def blank():
    logger.info("")
