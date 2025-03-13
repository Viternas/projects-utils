import json
import os
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any, Tuple, Union


class ConfigManager:
    def __init__(
            self,
            config_dir: Optional[str] = None,
            config_name: Optional[str] = None):
        self.config_dir = Path(
            config_dir or os.getenv(
                'CONFIG_DIR',
                '/default/config/dir'))
        self.config_name = config_name or os.getenv(
            'CONFIG_NAME', 'config.json')

        logger.info(f"Config Directory: {self.config_dir}")
        logger.info(f"Config Name: {self.config_name}")

    def open_config(self) -> Union[Tuple[bool, None], Tuple[bool, Path]]:
        config_path = self.config_dir / self.config_name

        if not self.config_dir.is_dir():
            logger.error(f"{self.config_dir} is not a valid directory")
            return False, None

        if not config_path.exists():
            logger.error(f"Config file {self.config_name} does not exist in {self.config_dir}")
            return False, None

        try:
            config_data = json.loads(config_path.read_text())
            logger.success(f"Config file {self.config_name} loaded successfully")
            return True, config_path
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON in {config_path}: {e}")
        except IOError as e:
            logger.error(f"Error reading file {config_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading config {config_path}: {e}")
        return False, None


if __name__ == '__main__':
    os.environ['CONFIG_DIR'] = './SRC'
    os.environ['CONFIG_NAME'] = 'config_encoded.json'

    config_manager = ConfigManager()
    success, config_data = config_manager.open_config()

    if success and config_data:
        print("Configuration Loaded:", config_data)
    else:
        print("Failed to load configuration")
