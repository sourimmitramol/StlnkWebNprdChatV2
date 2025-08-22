# utils/logger.py
import logging
from config import settings

def configure_logger() -> logging.Logger:
    logger = logging.getLogger("shipping_chatbot")
    logger.setLevel(settings.LOG_LEVEL)

    if not logger.handlers:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)

        # File handler
        fh = logging.FileHandler(settings.LOG_FILE)
        fh.setFormatter(formatter)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

logger = configure_logger()
