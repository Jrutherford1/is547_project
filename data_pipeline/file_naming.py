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

# Main execution for testing
if __name__ == "__main__":
    df = generate_names_csv()
    process_csv()