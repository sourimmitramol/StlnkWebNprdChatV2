# utils/logger.py
import logging
import sys

from config import settings


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("shipping_chatbot")
    logger.setLevel(settings.LOG_LEVEL)

    # Windows consoles often default to a legacy code page (e.g., cp1252/cp437)
    # which can crash logging when messages contain Unicode punctuation.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not logger.handlers:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)

        # File handler
        fh = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
        fh.setFormatter(formatter)

        # Console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


logger = configure_logger()
