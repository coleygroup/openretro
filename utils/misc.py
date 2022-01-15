import logging
import sys


def log_args(args, message: str):
    logging.info(message)
    for k, v in vars(args).items():
        logging.info(f"**** {k} = *{v}*")


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
