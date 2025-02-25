import os

from build_utils.create_config_class import CreateConfigClass

print(os.path.dirname(__file__))
cc = CreateConfigClass(package_dir=os.path.dirname(__file__), config_file_name='example.config_encoded.json').create_config_class()
from setup_master.master import Master
run = Master(working_dir=os.path.dirname(__file__), config_file_name='example.config_encoded.json')

