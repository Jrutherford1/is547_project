import os

def count_files_in_directory(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    print(f"Total number of files: {file_count}")

# Example usage
count_files_in_directory('data')