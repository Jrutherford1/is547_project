import os
import pandas as pd
import re
from datetime import datetime

# Define source directory and CSV output path
SOURCE_DIR = "./data/Committees"
CSV_OUTPUT = "./data/names.csv"

# Original date extraction function
def extract_date(filename):
    """Extracts a date from a filename. Handles:
       - YYYY-MM-DD, MM-DD-YYYY (including single-digit months/days)
       - Dates separated by periods (e.g., 02.03.2018)
       - Named months (e.g., "January 15, 2021" or "Jan 15 2021")
       - Returns a standardized YYYY-MM-DD format or 'unknown' if no date is found.
    """
    # Common date patterns to check (supports ., -, and _ as separators)
    date_patterns = [
        r'(\d{4}[-_.]\d{1,2}[-_.]\d{1,2})',  # YYYY-MM-DD, YYYY_MM_DD, YYYY.MM.DD
        r'(\d{1,2}[-_.]\d{1,2}[-_.]\d{4})',  # MM-DD-YYYY, MM_DD_YYYY, MM.DD.YYYY
        r'(\d{4}[-_.]\d{1,2})',              # YYYY-MM, YYYY_MM, YYYY.MM
    ]

    # Dictionary to map named months to numbers
    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
        "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
        "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
    }

    # First, try numeric date patterns
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(0).replace("_", "-").replace(".", "-")  # Normalize separators to hyphens
            try:
                return str(datetime.strptime(date_str, "%Y-%m-%d").date())  # Ensure valid date
            except ValueError:
                pass  # Skip invalid dates

    # Second, try named month formats (e.g., "January 15, 2021" or "Jan 15 2021")
    named_month_pattern = r'(?i)\b(' + '|'.join(month_map.keys()) + r')\s*(\d{1,2})[, ]*\s*(\d{4})\b'
    match = re.search(named_month_pattern, filename)

    if match:
        month_name, day, year = match.groups()
        month_num = month_map[month_name.lower()]
        return f"{year}-{month_num}-{day.zfill(2)}"  # Format as YYYY-MM-DD

    return "unknown"

# Creates a CSV with committee, type, original filename, extracted date, and proposed filename.
def generate_names_csv(source_dir=SOURCE_DIR, csv_output=CSV_OUTPUT):
    """Generates a CSV with file metadata and proposed names, excluding .DS_Store."""
    file_records = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file != ".DS_Store":  # Skip .DS_Store files
                source_path = os.path.join(root, file)
                parts = source_path.split(os.sep)
                committee = parts[-3] if len(parts) >= 4 else "Unknown"
                doc_type = parts[-2] if len(parts) >= 4 else "Unknown"
                extracted_date = extract_date(file)
                file_ext = os.path.splitext(file)[1]
                new_filename = f"{committee}_{doc_type}_{extracted_date}{file_ext}"
                file_records.append([committee, doc_type, file, extracted_date, new_filename])

    df_files = pd.DataFrame(file_records, columns=["Committee", "Document Type", "Original File Name", "Extracted Date", "Proposed File Name"])
    df_files.to_csv(csv_output, index=False)
    return df_files

# Loads the CSV, filters out .DS_Store, and saves a cleaned version.
def process_csv(csv_path=CSV_OUTPUT):
    """Processes the CSV to remove .DS_Store entries and saves a cleaned version."""
    df = pd.read_csv(csv_path)
    df_cleaned = df[df["Original File Name"] != ".DS_Store"]
    print(f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")
    cleaned_csv = csv_path.replace(".csv", "_cleaned.csv")
    df_cleaned.to_csv(cleaned_csv, index=False)
    return df_cleaned


def build_final_filenames(input_csv="data/manually_updated_committee_names.csv",
                          output_csv="data/final_updated_committee_names.csv"):
    """
    Builds final filenames using manually updated dates and saves to a new CSV.
    Replaces 'unknown' dates with dates from the Extracted Date column when available.
    """
    # Load the manually updated CSV
    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    required_columns = ["Committee", "Document Type", "Original File Name",
                        "Extracted Date", "Proposed File Name"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {input_csv}")

    # Function to update filename with new date
    def update_filename(row):
        date = row["Extracted Date"]
        # If date was 'unknown' but now has a valid date format
        if ("unknown" in str(row["Proposed File Name"]).lower() and
                date != "unknown" and pd.notna(date)):
            # Extract file extension
            file_ext = os.path.splitext(row["Original File Name"])[1]
            # Build new filename with updated date
            return f"{row['Committee']}_{row['Document Type']}_{date}{file_ext}"
        return row["Proposed File Name"]

    # Apply the update function to create final filenames
    df["Final File Name"] = df.apply(update_filename, axis=1)

    # Save the updated DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Final filenames saved to {output_csv}")
    print(f"Rows processed: {len(df)}")

    return df


def verify_folder_structure(processed_dir="data/Processed_Committees",
                            final_csv="data/final_updated_committee_names.csv"):
    """
    Verifies that the folder structure in final_updated_committee_names.csv matches
    the actual structure in Processed_Committees.
    Returns True if all matches, False if any mismatches are found, and prints details.
    """

    # Load the final updated CSV
    df = pd.read_csv(final_csv)

    # Ensure required columns exist
    required_columns = ["Committee", "Document Type", "Original File Name"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {final_csv}")

    all_match = True
    mismatch_count = 0

    # Check each row in the CSV
    for index, row in df.iterrows():
        # Construct the expected path based on CSV data
        expected_rel_path = os.path.join(
            row["Committee"],
            row["Document Type"],
            row["Original File Name"]
        )
        expected_full_path = os.path.join(processed_dir, expected_rel_path)

        # Check if the committee directory exists
        committee_path = os.path.join(processed_dir, row["Committee"])
        if not os.path.isdir(committee_path):
            print(f"Error at row {index}: Committee directory not found: {committee_path}")
            all_match = False
            mismatch_count += 1
            continue

        # Check if the document type subdirectory exists
        doc_type_path = os.path.join(committee_path, row["Document Type"])
        if not os.path.isdir(doc_type_path):
            print(f"Error at row {index}: Document Type directory not found: {doc_type_path}")
            all_match = False
            mismatch_count += 1
            continue

        # Check if the original file exists
        if not os.path.isfile(expected_full_path):
            print(f"Error at row {index}: File not found at expected path: {expected_full_path}")
            all_match = False
            mismatch_count += 1

    # Summary report
    if all_match:
        print(f"Verification complete: All {len(df)} entries match the folder structure in {processed_dir}")
    else:
        print(f"Verification complete: Found {mismatch_count} mismatches out of {len(df)} entries")

    return all_match



def rename_processed_files(processed_dir="data/Processed_Committees",
                           final_csv="data/final_updated_committee_names.csv"):
    """
    Renames files in Processed_Committees using final filenames from the CSV.
    Preserves the original folder structure and only modifies Processed_Committees.
    """

    # Load the final updated CSV
    df = pd.read_csv(final_csv)

    # Ensure required columns exist
    required_columns = ["Original File Name", "Final File Name"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {final_csv}")

    # Create a mapping of original paths to final filenames
    rename_map = {}
    for _, row in df.iterrows():
        # Reconstruct the original path in Processed_Committees
        original_rel_path = os.path.join(
            row["Committee"],
            row["Document Type"],
            row["Original File Name"]
        )
        original_full_path = os.path.join(processed_dir, original_rel_path)

        # Get the final filename (just the filename, not full path)
        final_filename = row["Final File Name"]

        # Construct the final full path (keeping same directory structure)
        final_full_path = os.path.join(
            processed_dir,
            row["Committee"],
            row["Document Type"],
            final_filename
        )

        rename_map[original_full_path] = final_full_path

    # Perform the renaming
    renamed_count = 0
    for original_path, new_path in rename_map.items():
        if os.path.exists(original_path):
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(original_path, new_path)
            renamed_count += 1
            print(f"Renamed: {original_path} -> {new_path}")
        else:
            print(f"Warning: File not found for renaming: {original_path}")

    print(f"Total files renamed: {renamed_count}")
    return renamed_count


