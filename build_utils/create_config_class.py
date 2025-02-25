import json
import os
import subprocess
import pathlib
import utils

class CreateConfigClass(object):
    def __init__(self, package_dir: str, config_file_name: str):
        """
            Initializes CreateConfigClass with the provided package directory and config file name.

            Args:
                package_dir (str): Directory path where the config file is located
                config_file_name (str): Name of the config file (e.g., 'config.json')
            """
        self.package_dir = package_dir
        config_file = pathlib.Path(self.package_dir) / config_file_name

        with open(config_file, 'r') as file:
            self.config_file = json.load(file)
            self.config_file_name = config_file
            if package_dir is not None:
                self.config_file_location = package_dir
            else:
                self.config_file_location = self.config_file_name

    def create_config_class(self):
        """
        Generates the Config.py file based on the provided config_file.
        """
        config_py = pathlib.Path(os.path.dirname(utils.__file__)) / 'config/config.py'
        with open(config_py, 'w') as file:
            file.writelines("import json")
            file.writelines("\n\n\nclass Config(object): \n")
            if self.config_file_location != self.config_file_name:
                file.writelines(
                    "    def __init__(self, kubernetes: bool = None, config_file: str = None):\n")
                file.writelines("        if kubernetes:\n")
                file.writelines("            self.k8_mode()\n")
                file.writelines("        else:\n")
                file.writelines(
                    "            if not config_file:\n")
                file.writelines(
                    "                with open('%s') as files:\n" %
                    self.config_file_name)
                file.writelines(
                    "                    self.data = json.load(files)\n")
                file.writelines(
                    "            else:\n")
                file.writelines(
                    "                with open(config_file) as files:\n")
                file.writelines(
                    "                    self.data = json.load(files)\n")
                file.writelines("    def k8_mode(self):\n")
                file.writelines(
                    "        with open('%s') as files:\n" %
                    self.config_file_location)
                file.writelines("            file = files.read()\n")
                file.writelines("        with open(file) as files:\n")
                file.writelines("            self.data = json.load(files)\n")
                file.writelines("            files.close()\n")
                file.writelines("        return\n\n")
            else:
                file.writelines("    def __init__(self, config_file):\n")
                file.writelines("            self.config_file = config_file\n")
                file.writelines(
                    "            with open(self.config_file) as files:\n"
                )
                file.writelines(
                    "                self.data = json.load(files)\n")

            for main_key in self.config_file:
                def_string = "    def return_config_%s(self, " % main_key
                for value in self.config_file[main_key]:
                    def_string += "%s_%s: bool = None, " % (main_key, value)
                def_string = def_string[:-2] + "):\n"
                file.writelines(def_string)
                file.writelines("        db = self.data['%s']\n" % main_key)
                for value in self.config_file[main_key]:
                    file.writelines("        if %s_%s:\n" % (main_key, value))
                    file.writelines("            return db['%s']\n" % value)
                file.writelines("\n")
            file.writelines("\nif __name__ == '__main__':\n")
            file.writelines("    print('On Main')")
        subprocess.run(['python',
                        '-m',
                        'autopep8',
                        '--in-place',
                        '--aggressive',
                        '--aggressive',
                        config_py],
                       check=True)
#


if __name__ == '__main__':
    # Example usage
    import utils.config
    package_dir = os.path.dirname(utils.config.__file__)
    CreateConfigClass(package_dir=package_dir,
                      config_file_name='example.config_encoded.json'
                      ).create_config_class()
