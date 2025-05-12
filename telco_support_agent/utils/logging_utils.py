import logging


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Suppress py4j messages - CRITICAL level
    logging.getLogger('py4j').setLevel(logging.CRITICAL)


def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger, typically __name__

    Returns:
        logging.Logger: A configured logger instance
    """
    return logging.getLogger(name)
