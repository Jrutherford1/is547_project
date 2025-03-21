import os

#Get simple file count to check accuracy, I know it is 2204.  Excludes mac .DS_Store files
def count_files(directory):
    file_count = 0

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            # Exclude .DS_Store and any hidden system files
            if not file_name.startswith("."):
                file_count += 1

    print(f"Total number of valid files (excluding system files): {file_count}")


# Get file types and count of each type for scoping and planning
def find_file_types(directory):
    file_type_counts = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1]
            if file_extension in file_type_counts:
                file_type_counts[file_extension] += 1
            else:
                file_type_counts[file_extension] = 1

    return file_type_counts

# Get list of committees and count of each committee for scoping and planning
def list_committees_and_count(directory):
    committee_count = 0
    committees_path = os.path.join(directory, 'Committees')

    if os.path.exists(committees_path):
        for committee_name in os.listdir(committees_path):
            committee_path = os.path.join(committees_path, committee_name)
            if os.path.isdir(committee_path):
                print (committee_name)
                committee_count += 1

    print(f"Total number of committees: {committee_count}")

# List 100 files for review of output
def list_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_count <100:
                print(f"File: {os.path.join(root, file_name)}")
                file_count += 1
            else:
                break
        if file_count >= 100:
            break



