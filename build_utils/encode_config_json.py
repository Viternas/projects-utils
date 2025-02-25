import base64
import json
import os
import pathlib


class JsonFormatter:
    def __init__(
            self,
            package_dir: str,
            config_file,
            encode: bool = None,
            encrypt: bool = None):
        self.package_dir = package_dir
        self.config_file = pathlib.Path(package_dir) / config_file
        self.encode = encode
        self.encrypt = encrypt
        if self.encode:
            self.encoder()

    def encoder(self):
        with open(self.config_file) as files:
            file = json.load(files)
        encoded_json = self._encode_dict(file)

        self.save_encoded_file(encoded_json)

    def _encode_dict(self, data):
        encoded_dict = {}

        for key, value in data.items():
            if isinstance(value, dict):
                encoded_dict[key] = self._encode_dict(value)
            elif isinstance(value, str):
                text_bytes = value.encode('utf-8')
                base64_bytes = base64.b64encode(text_bytes)
                base64_text = base64_bytes.decode('utf-8')
                encoded_dict[key] = base64_text
            else:
                encoded_dict[key] = value

        return encoded_dict

    def save_encoded_file(self, final_json):
        base_file_name = os.path.splitext(self.config_file)[0]
        print(base_file_name)
        base_file_name = base_file_name.split('unencoded_')[1]
        encoded_file_name = f"{base_file_name}_encoded.json"
        output_file = pathlib.Path(self.package_dir) / encoded_file_name
        with open(output_file, 'w') as outfile:
            json.dump(final_json, outfile, indent=4)
        print(f"Encoded JSON saved to {encoded_file_name} @ {self.package_dir}")


if __name__ == '__main__':
    import utils.config
    package_dir = os.path.dirname(utils.config.__file__)
    print(package_dir)
    run = JsonFormatter(
        package_dir=package_dir,
        config_file='unencoded_example.config.json',
        encode=True)
