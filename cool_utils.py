import json


# import the config file
def load_config(config_path: str = "./config.json") -> dict:
    """Load configuration from JSON file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to "config.json".

    Returns:
        dict: Configuration dictionary.
    """

    with open(config_path, "r") as f:
        return json.load(f)


def print_red(text: str) -> None:
    """Print text in red color.

    Args:
        text (str): The text to print
    """
    print(f"\033[91m{text}\033[0m")


def print_green(text: str) -> None:
    """Print text in green color.

    Args:
        text (str): The text to print
    """
    print(f"\033[92m{text}\033[0m")


def print_blue(text: str) -> None:
    """Print text in blue color.

    Args:
        text (str): The text to print
    """
    print(f"\033[94m{text}\033[0m")
