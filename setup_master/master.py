from loguru import logger
from typing import Optional
import os

from sqlalchemy.testing.plugin.plugin_base import config

from setup_master.config_manager import ConfigManager

import utils
from utils.exceptions.exceptions import *
from utils.config.config import Config as CF
from utils.database.database_driver import DatabaseDriver as DB
from utils.secrets.secrets import Encrypt as SE
from utils.ai.ai_driver import AIDriver as AI


class Master:
    def __init__(self, working_dir: str=None, config_file_directory: str=None, config_file_name: str=None):
        self.working_directory = working_dir
        self.config_file_directory = config_file_directory
        self.config_file_name = config_file_name
        self.set_working_directory()
        logger.info("STARTING MASTER")

        self.CF: Optional[CF] = None
        self.SE: Optional[SE] = None
        self.DB: Optional[DB] = None
        self.AI: Optional[AI] = None

        self.load_configuration()
        self.temp_dir = "TEMP_STORE"

        self.set_security()
        self.set_database_manager()
        self.set_ai()

        self.global_process_limit = 25

    def load_configuration(self):
        if not self.config_file_directory:
            config_dir = os.path.dirname(
                os.path.abspath(utils.config.__file__))
            config_result = ConfigManager(
                config_dir=config_dir,
                config_name=self.config_file_name).open_config()
        else:
            config_result = ConfigManager(
                config_dir=self.config_file_directory,
                config_name=self.config_file_name).open_config()

        if isinstance(
                config_result,
                tuple) and config_result[0] and config_result[1]:
            self.CF = CF(kubernetes=False, config_file=config_result[1])
        else:
            raise ConfigurationError("Configuration could not be loaded")

    def set_security(self):
        try:
            logger.info("Setting up security...")
            key_location = os.getenv('KEY_LOCATION', '')
            self.SE = SE(
                key_location=SE.base64_decode_string(
                    encoded_string=key_location), secret_key=SE.base64_decode_string(
                    encoded_string=self.CF.return_config_secrets(
                        secrets_fernet_key=True)), salt_key=SE.base64_decode_string(
                    encoded_string=self.CF.return_config_secrets(
                        secrets_salt_key=True)), )
            logger.success("SECRETS initialized")
        except Exception as e:
            logger.error(f"SECRETS error: {e}")
            raise SecurityError(f"Failed to initialize secrets: {e}")

    def set_database_manager(self):
        try:
            logger.info("Setting up the database manager...")
            self.DB = DB(
                user=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_database(
                        database_user=True)), password=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_database(
                        database_password=True)), host=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_database(
                        database_host=True)), port=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_database(
                        database_port=True)), database=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_database(
                        database_database=True)), maxcon=self.CF.return_config_database(
                    database_max_connection=True), mincon=self.CF.return_config_database(
                    database_min_connection=True))
            logger.success("Database manager setup successfully.")
        except Exception as e:
            logger.error(f"Failed to set up the database manager: {e}")
            raise DatabaseError(f"Failed to set up the database manager: {e}")

    def set_ai(self):
        try:
            logger.info("Setting up the GPT class...")
            self.AI = AI(
                OPEN_ROUTER_KEY=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_ai_keys(
                        ai_keys_OpenRouter_Key=True)), OLLAMA_KEY=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_ai_keys(
                        ai_keys_Ollama_Key=True)), OPEN_AI_KEY=self.SE.base64_decode_string(
                    encoded_string=self.CF.return_config_ai_keys(
                        ai_keys_OpenAi_key=True)))
            logger.success("GPT initialized")
        except Exception as e:
            logger.error(f"GPT initialization error: {e}")
            raise GPTError(f"Failed to initialize GPT: {e}")

    def set_working_directory(self):
        try:
            logger.info("Setting working directory")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if self.working_directory is None:
                os.chdir(script_dir)
                logger.success(f"Working directory set to {script_dir}")
            else:
                os.chdir(self.working_directory)
                logger.success(
                    f"Working directory set to {
                        self.working_directory}")
        except Exception as e:
            logger.error(f"Failed to set working directory: {e}")
            raise


if __name__ == '__main__':
    test_run = Master(working_dir='', config_file_name='example.config_encoded.json')
