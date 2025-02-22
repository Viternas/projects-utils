# Project Utils

A collection of Python utilities for database operations, AI integration, secure configuration management, and encryption.

## Features

- **Database Management**: Connection pooling, query builder, and safe SQL operations
- **AI Integration**: Support for OpenAI, OpenRouter, and Ollama with unified interface
- **Security**: Encryption, hashing, and secure configuration handling
- **Configuration**: Flexible JSON-based config management with encoding support

## Installation

```bash
pip install project-utils
```

## Config.py creation
```bazaar
python3 CreateConfig.py
```

## Quick Start

```python
from project_utils import Master

# Initialize with default configuration
master = Master()

# Database operations
result = master.DB.execute("SELECT version()")

# AI operations
response = master.AI.chat_with_gpt("Hello, how are you?")

# Encryption
encrypted = master.SE.encrypt_message("sensitive data")
```

## Configuration

Create a configuration file `config.json`:

```json
{
    "database": {
        "host": "localhost",
        "port": "5432",
        "database": "mydb",
        "user": "user",
        "password": "pass",
        "max_connection": 10,
        "min_connection": 1
    },
    "ai_keys": {
        "OpenRouter_Key": "your-key",
        "Ollama_Key": "your-key",
        "OpenAi_key": "your-key"
    }
}
```

## Components

### Database Driver
```python
from project_utils import DatabaseDriver

db = DatabaseDriver(
    user="user",
    password="pass",
    host="localhost",
    port="5432",
    database="mydb"
)
```

### AI Driver
```python
from project_utils import AIDriver

ai = AIDriver(
    OPEN_ROUTER_KEY="key",
    OPEN_AI_KEY="key",
    OLLAMA_KEY="key"
)
```

### Security
```python
from project_utils import Encrypt

encrypt = Encrypt(secret_key="your-key", salt_key="your-salt")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Security

For encryption and security features, please ensure you follow best practices for key management. Never commit sensitive keys to version control.

## Requirements

- Python >= 3.8
- Dependencies listed in setup.py