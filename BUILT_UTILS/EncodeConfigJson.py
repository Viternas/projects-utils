import base64
import json
import os
import pathlib


class JsonFormatter:
    def __init__(self, package_dir: str, config_file, encode: bool = None, encrypt: bool = None):
        self.config_file = pathlib.Path(package_dir) / config_file
        self.encode = encode
        self.encrypt = encrypt
        if self.encode:
            self.encoder()

    def encoder(self):
        with open(self.config_file) as files:
            file = json.load(files)

        # Start encoding the JSON structure
        encoded_json = self._encode_dict(file)

        # Save the processed JSON to a new file with the _encoded suffix
        self.save_encoded_file(encoded_json)

    def _encode_dict(self, data):
        """Recursively encode a dictionary."""
        encoded_dict = {}

        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively encode nested dictionaries
                encoded_dict[key] = self._encode_dict(value)
            elif isinstance(value, str):
                # Encode strings to base64
                text_bytes = value.encode('utf-8')
                base64_bytes = base64.b64encode(text_bytes)
                base64_text = base64_bytes.decode('utf-8')
                encoded_dict[key] = base64_text
            else:
                # Keep other data types as they are
                encoded_dict[key] = value

        return encoded_dict

    def save_encoded_file(self, final_json):
        base_file_name = os.path.splitext(self.config_file)[0]
        encoded_file_name = f"{base_file_name}_encoded.json"
        with open(encoded_file_name, 'w') as outfile:
            json.dump(final_json, outfile, indent=4)
        print(f"Encoded JSON saved to {encoded_file_name}")


if __name__ == '__main__':
    import SRC.UTILS.CONFIG
    package_dir = os.path.dirname(SRC.UTILS.CONFIG.__file__)
    run = JsonFormatter(package_dir=package_dir, config_file='unencoded_example.config.json', encode=True)
