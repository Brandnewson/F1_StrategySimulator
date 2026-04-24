import json
from typing import Dict
#  Function to load configuration into nested dictionaries
def load_config(file_path: str) -> Dict:
    """Load and parse the configuration file into nested dictionaries."""
    with open(file_path, 'r') as file:
        config_data = json.load(file)

    # Preserve all config keys so new experiment interfaces can be added
    # without having to change this loader each time.
    parsed_config = dict(config_data)

    return parsed_config
