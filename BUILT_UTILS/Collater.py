import os


def find_unique_py_files(start_dir, exclude_folders=None):
    """
    Recursively finds all unique Python files (.py) in nested directories,
    excluding specified folders.

    :param start_dir: The directory to start searching from.
    :param exclude_folders: List of folder names to exclude from the search.
    :return: A list of unique paths to Python files found.
    """
    if exclude_folders is None:
        exclude_folders = []

    py_files = {}
    for root, dirs, files in os.walk(start_dir):
        # Remove excluded folders from dirs list to prevent recursing into them
        dirs[:] = [d for d in dirs if d not in exclude_folders]

        for file in files:
            if file.endswith('.py'):
                file_name = os.path.basename(file)  # Get just the file name
                file_path = os.path.join(root, file)
                # Keep './' prefixed version if it exists, otherwise take any
                if file_name not in py_files or file_path.startswith('./'):
                    py_files[file_name] = file_path
    return list(py_files.values())


def write_files_to_text(py_files, output_file):
    """
    Reads and writes the content of all Python files to a text file.

    :param py_files: A list of paths to Python files.
    :param output_file: The file to write all the contents to.
    """
    with open(output_file, 'w') as outfile:
        for py_file in py_files:
            with open(py_file, 'r') as infile:
                outfile.write(f"\n\n--- {py_file} ---\n\n")
                outfile.write(infile.read())
                outfile.write("\n\n")


if __name__ == "__main__":
    # Replace '.' with the path of the directory you want to search in
    start_directory = '.'
    output_text_file = 'Agents.txt'

    # Specify folders to exclude
    folders_to_exclude = ['venv', '__pycache__', '.git', 'ThreadFactory', 'SPECIALITY', 'ProjectFriday.egg-info']  # Add your folders here

    # Find all unique Python files, excluding specified folders
    unique_python_files = find_unique_py_files(start_directory, folders_to_exclude)

    for item in unique_python_files:
        print(item)

    # Write the contents of all unique Python files into a single text file
    write_files_to_text(unique_python_files, output_text_file)

    print(f"All unique Python files have been written to {output_text_file}.")