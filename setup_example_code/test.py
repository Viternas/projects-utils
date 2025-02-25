import os

from build_utils.create_config_class import CreateConfigClass
from build_utils.encode_config_json import JsonFormatter

print(os.path.dirname(__file__))
ec = JsonFormatter(package_dir=os.path.dirname(__file__), config_file='unencoded_example.config.json', encode=True)
cc = CreateConfigClass(package_dir=os.path.dirname(__file__), config_file_name='example.config_encoded.json').create_config_class()
from setup_master.master import Master
run = Master(working_dir=os.path.dirname(__file__), config_file_name='example.config_encoded.json')

