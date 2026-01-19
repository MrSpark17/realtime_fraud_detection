import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "app_config.yaml")

with open(CONFIG_PATH, "r") as f:
    _raw_cfg = yaml.safe_load(f)

def get_config():
    return _raw_cfg
