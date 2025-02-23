"""Functions shared across"""

import yaml
import logging


def load_config(config_path="config.yml"):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def create_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger
