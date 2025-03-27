import os
import shutil

# Define source and destination directories
SOURCE_DIR = "./data/Committees"
OUTPUT_DIR = "./data/Processed_Committees"

# Ensures that the output directory exists.
def ensure_output_directory(output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Copies files from SOURCE_DIR to OUTPUT_DIR while preserving folder structure and skipping .DS_Store.
def copy_files(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_path = os.path.join(root, file)
            if file == ".DS_Store":
                print(f"Deleting: {source_path}")
                os.remove(source_path)
            else:
                rel_path = os.path.relpath(source_path, source_dir)
                dest_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)

# Lists a sample of 100 files for review, skipping .DS_Store.
def list_files(directory, max_files=100):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_count < max_files and file_name != ".DS_Store":
                print(f"File: {os.path.join(root, file_name)}")
                file_count += 1
            if file_count >= max_files:
                return

