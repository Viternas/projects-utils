from setuptools import setup, find_packages

setup(
    name="project_utils",
    version="0.1.0",
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires=[
        "setuptools",
        "autopep8",
        "psycopg2>=2.9.10",
        "sqlalchemy>=2.0.38",
        "cryptography>=3.4.0",
        "passlib>=1.7.4",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    python_requires=">=3.8",
    description="Agent utility functions for database, AI, config and security operations",
    package_data={
        'utils.config': ['*.json'],  # Updated package data path
    },
    include_package_data=True,
)
