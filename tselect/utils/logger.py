import logging

def setup_logger(level="INFO"):
    logger = logging.getLogger("tselect")

    if logger.handlers:
        return logger

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    logger.setLevel(level_map.get(level.upper(), logging.INFO))

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
