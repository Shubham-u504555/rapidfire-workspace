import logging, os, sys
def get_logger(name: str = "rapidfire"):
    logger = logging.getLogger(name)
    if logger.handlers: return logger
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger
