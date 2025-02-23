"""Functions shared across"""

import yaml


def load_config(config_path="config.yml"):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
