import os

from build_utils.create_config_class import CreateConfigClass
from build_utils.encode_config_json import JsonFormatter

#print(os.path.dirname(__file__))
#ec = JsonFormatter(package_dir=os.path.dirname(__file__), config_file='unencoded_test.config.json', encode=True)
#cc = CreateConfigClass(package_dir=os.path.dirname(__file__), config_file_name='test.config_encoded.json').create_config_class()
from setup_master.master import Master
from utils.ai.ai_enums import Models

run = Master(working_dir=os.path.dirname(__file__), config_file_name='test.config_encoded.json', config_file_directory=os.path.dirname(__file__))

run.AI.model = Models.GEMMA2_9B.model_id
a = run.AI.ollama_chat('this is a test')
print(a)

