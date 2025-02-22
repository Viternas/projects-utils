import base64
import os
from cryptography.fernet import Fernet
import http.client
import json
import hashlib
import math
from passlib.context import CryptContext


class Encrypt(object):
    def __init__(self, key_location: str = None,
                 conn: str = None,
                 salt_key: str = None,
                 secret_key: str = None,
                 ):

        self.conn = ""#http.client.HTTPSConnection(conn)
        self.salt_key = salt_key
        self.secret_key = secret_key
        self.pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

    """
     @Notice: This function will generate an encryption secret key
     @Dev:    We user the Fernet generate_key method to generate a key and store in a secret.key file
    """

    def generate_key(self):
        with open("./src/secret.key", "wb") as key_file:
            key_file.write(Fernet.generate_key())

    """
     @Notice: This function will load an existing encryption key from a file
     @Dev:    We open a file and return its content
    """

    def load_key(self, key_path=None):
        if key_path is None:
            return open("./src/secret.key", "rb").read()
        return open(key_path, "rb").read()

    """
     @Notice: This function will encrypt a string using the loaded encryption key
     @Dev:    We use the Fernet encrypt method and the loaded encryption key to encrypt a string
    """

    def encrypt_message(self, message):
        return Fernet(self.secret_key).encrypt(str(message).encode()).decode()

    """
     @Notice: This function will decrypt a string using the loaded encryption key
     @Dev:    We use the Fernet decrypt method and the loaded encryption key to decrypt a string
    """

    def decrypt_message(self, encrypted_message):
        return Fernet(self.secret_key).decrypt(encrypted_message).decode()

    """
     @Notice: This function will generate a sha3 row hash
     @Dev:    We concatenate the arguments string list to a string, hash it and 32 bits salt it, then we salt it again
     using the salt.key
    """

    def hash_list(self, args_list: list):
        concatenated_row = "".join(str(e) for e in args_list)
        obj_sha3_256 = hashlib.sha3_256(concatenated_row.encode())
        hash_salt = "".join(
            str(letter) for index,
            letter in enumerate(concatenated_row) if index %
            2 == 0)
        if len(hash_salt) > 32:
            hash_salt = hash_salt[:32]
        elif len(hash_salt) < 32:
            hash_salt = hash_salt * math.floor(32 / len(hash_salt))
            hash_salt = hash_salt + hash_salt[:(32 % len(hash_salt))]
        obj_sha3_256.update(hash_salt.encode())
        obj_sha3_256.update(self.salt_key.encode())
        return obj_sha3_256.hexdigest()

    def hash_password(self, password: str):
        return self.pwd_context.hash(password)

    def verify_password(
            self,
            plain_password: str,
            hashed_password: str) -> bool:
        """
        Verifies a password against its hash.

        Parameters:
        - plain_password (str): The plain text password to verify.
        - hashed_password (str): The hashed password for comparison.

        Returns:
        - bool: True if the password matches the hash, False otherwise.
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def base64_encode_string(unencoded_string: str):
        return base64.b64encode(unencoded_string.encode()).decode()

    @staticmethod
    def base64_decode_string(encoded_string: str):
        return base64.b64decode(encoded_string).decode()


if __name__ == "__main__":
    test = Encrypt()
    print(test.base64_decode_string(encoded_string='eyd0YXNrJzogJzFiMTU1MzgwODk5NjAxZmMyOGZiZGY1ZjEyMDYxZmY1M2YzYTBhODIwMTA3NzA0MWEyMjFmY2ViNTcyM2EzYmUnfQ=='))