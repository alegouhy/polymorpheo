import logging
from typing import Optional

# Ensure library loggers have a NullHandler by default so importing the
# library doesn't configure global logging for applications that use it.
logging.getLogger("contours2mesh").addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger. Use library consumers' names (module __name__).

    Example: ``logger = get_logger(__name__)`` in modules.
    """
    return logging.getLogger(name or "contours2mesh")


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """Configure root logging for applications using this library.

    This will only configure logging if the root logger has no handlers.
    """
    root = logging.getLogger()
    if not root.handlers:
        fmt = fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=level, format=fmt)
