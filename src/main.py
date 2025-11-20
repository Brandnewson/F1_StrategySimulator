import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.configUtils import load_config
from src.track import load_track
from src.simulator import init_simulator

# Load configuration
config = load_config("config.json")
print("Loaded Configuration:")
print(json.dumps(config, indent=2, sort_keys=True))

# Load track
track = load_track(config)

# Initialise simulator
race_state = init_simulator(config, track)

