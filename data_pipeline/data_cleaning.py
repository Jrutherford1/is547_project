import os
import shutil

# Define source and destination directories
SOURCE_DIR = "./data/Committees"
OUTPUT_DIR = "./data/Processed_Committees"

# Ensures that the output directory exists so I don't puke files all over.
def ensure_output_directory(output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Copies files from SOURCE_DIR to OUTPUT_DIR while preserving folder structure and skipping .DS_Store.
def copy_files(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR):
    for root, dirs, files in os.walk(source_dir):
        # First, create all subdirectories (including empty ones).
        for d in dirs:
            new_dir = os.path.join(output_dir, os.path.relpath(os.path.join(root, d), source_dir))
            os.makedirs(new_dir, exist_ok=True)

        # Then copy all files, delete DS_Store files (these plagued me at first).
        for file in files:
            if file == ".DS_Store":
                source_path = os.path.join(root, file)
                print(f"Deleting: {source_path}")
                os.remove(source_path)
            else:
                source_path = os.path.join(root, file)
                rel_path = os.path.relpath(source_path, source_dir)
                dest_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)

# Lists a sample of 100 files for review, skipping .DS_Store (anything to get rid of them).
def list_files(directory, max_files=100):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_count < max_files and file_name != ".DS_Store":
                print(f"File: {os.path.join(root, file_name)}")
                file_count += 1
            if file_count >= max_files:
                return


def clean_ds_store_files(base_dir="data/Processed_Committees"):
    """
    Finds and removes any .DS_Store files contaminating the processed committees directory.

    Args:
        base_dir: Directory to clean (default: "data/Processed_Committees")

    Returns:
        int: Number of .DS_Store files found and removed
    """
    removed_count = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == ".DS_Store":
                ds_store_path = os.path.join(root, file)
                print(f"Found and removing: {ds_store_path}")
                try:
                    os.remove(ds_store_path)
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {ds_store_path}: {e}")

    if removed_count == 0:
        print("✅ No .DS_Store files found - directory is clean!")
    else:
        print(f"🧹 Removed {removed_count} .DS_Store file(s)")

    return removed_count


# Run the cleanup
if __name__ == "__main__":
    clean_ds_store_files()