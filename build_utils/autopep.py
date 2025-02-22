import pathlib
import subprocess


def auto_pepper(root_dir: str):
    path = pathlib.Path(root_dir)

    python_files = [
        file for file in path.rglob("*.py")
        if ".venv" not in file.parts
    ]

    # Print each Python file path
    for file in python_files:
        subprocess.run(['python', '-m', 'autopep8', '--in-place',
                       '--aggressive', '--aggressive', file], check=True)


# Usage
auto_pepper(root_dir='../')
