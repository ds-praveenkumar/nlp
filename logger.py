import logging


def get_basic_logger():
    logging.basicConfig(level=logging.DEBUG, format= "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")
    logger = logging.getLogger()
    return logger
