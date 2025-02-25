import json


class Config(object):
    def __init__(self, kubernetes: bool = None, config_file: str = None):
        if kubernetes:
            self.k8_mode()
        else:
            if not config_file:
                with open('/setup_example_code/example.config_encoded.json') as files:
                    self.data = json.load(files)
            else:
                with open(config_file) as files:
                    self.data = json.load(files)

    def k8_mode(self):
        with open('/setup_example_code') as files:
            file = files.read()
        with open(file) as files:
            self.data = json.load(files)
            files.close()
        return

    def return_config_database(
            self,
            database_encoded: bool = None,
            database_database: bool = None,
            database_user: bool = None,
            database_password: bool = None,
            database_host: bool = None,
            database_port: bool = None,
            database_max_connection: bool = None,
            database_min_connection: bool = None):
        db = self.data['database']
        if database_encoded:
            return db['encoded']
        if database_database:
            return db['database']
        if database_user:
            return db['user']
        if database_password:
            return db['password']
        if database_host:
            return db['host']
        if database_port:
            return db['port']
        if database_max_connection:
            return db['max_connection']
        if database_min_connection:
            return db['min_connection']

    def return_config_secrets(
            self,
            secrets_encoded: bool = None,
            secrets_fernet_key: bool = None,
            secrets_salt_key: bool = None,
            secrets_conn: bool = None):
        db = self.data['secrets']
        if secrets_encoded:
            return db['encoded']
        if secrets_fernet_key:
            return db['fernet_key']
        if secrets_salt_key:
            return db['salt_key']
        if secrets_conn:
            return db['conn']

    def return_config_ai_keys(
            self,
            ai_keys_encoded: bool = None,
            ai_keys_OpenAi_key: bool = None,
            ai_keys_OpenRouter_Key: bool = None,
            ai_keys_Ollama_Key: bool = None):
        db = self.data['ai_keys']
        if ai_keys_encoded:
            return db['encoded']
        if ai_keys_OpenAi_key:
            return db['OpenAi_key']
        if ai_keys_OpenRouter_Key:
            return db['OpenRouter_Key']
        if ai_keys_Ollama_Key:
            return db['Ollama_Key']

    def return_config_orchestration_engine(
            self,
            orchestration_engine_host: bool = None,
            orchestration_engine_port: bool = None,
            orchestration_engine_api_key: bool = None):
        db = self.data['orchestration_engine']
        if orchestration_engine_host:
            return db['host']
        if orchestration_engine_port:
            return db['port']
        if orchestration_engine_api_key:
            return db['api_key']


if __name__ == '__main__':
    print('On Main')
