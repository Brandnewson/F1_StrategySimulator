import json
from typing import Dict
#  Function to load configuration into nested dictionaries
def load_config(file_path: str) -> Dict:
    """Load and parse the configuration file into nested dictionaries."""
    with open(file_path, 'r') as file:
        config_data = json.load(file)

    parsed_config = {
        "track": config_data.get("track", {}),
        "simulator": config_data.get("simulator", {}),
        "competitors": config_data.get("competitors", {})
    }

    return parsed_config