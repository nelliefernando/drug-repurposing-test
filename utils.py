import yaml


def get_config(config_file):
    """Read config file.

    Args:
        config_file (str): path to config file.

    Returns:
        dict: config settings.
    """
    with open(config_file, "r") as config:
        config = yaml.safe_load(config)
    return config
