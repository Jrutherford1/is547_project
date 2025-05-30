import os
import json
import hashlib
import pandas as pd
from collections import defaultdict


# Calculate hashes.  NGL, I don't know exactly how this part works.
def calculate_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def enhance_metadata(
    csv_path="data/final_updated_committee_names.csv",  # Changed to use final CSV
    base_dir="data/Processed_Committees",
    skip_existing=False  # Changed to False to always regenerate
):
    # Load and normalize the CSV
    df = pd.read_csv(csv_path)
    df.rename(columns={
        'Final File Name': 'final_filename',  # Changed from 'Proposed File Name'
        'Committee': 'committee',
        'Document Type': 'type',
        'Extracted Date': 'date',
        'Original File Name': 'original_name'
    }, inplace=True)
    df['final_filename'] = df['final_filename'].str.strip()

    # Stats tracking for human error checking
    file_type_counts = defaultdict(int)
    skipped = 0
    updated = 0
    created = 0

    # concatenate the committee and type columns to create a relative path
    for _, row in df.iterrows():
        committee = row['committee'].strip()
        doc_type = row['type'].strip()
        filename = row['final_filename'].strip()

        rel_path = os.path.join(committee, doc_type, filename)
        full_path = os.path.join(base_dir, rel_path)

        if not os.path.isfile(full_path):
            print(f"Missing: {rel_path}")
            continue

        # Build
        file_type = os.path.splitext(filename)[-1].lower()
        # Count file types
        file_type_counts[file_type] += 1

        checksum = calculate_sha256(full_path)
        base_name = os.path.splitext(filename)[0]
        folder_path = os.path.dirname(full_path)
        # Create the JSON path/name
        json_path = os.path.join(folder_path, base_name + ".json")

        # Track if we're creating a new file or updating an existing one
        if os.path.exists(json_path):
            updated += 1
        else:
            created += 1

        # Use JSON-LD format with Schema.org vocab
        metadata = {
            "@context": {
                "@vocab": "http://schema.org/",
                "checksum": "https://schema.org/checksum",
                "algorithm": "https://schema.org/algorithm",
                "value": "https://schema.org/value"
            },
            "@type": "CreativeWork",
            "name": filename,
            "creator": {
                "@type": "Organization",
                "name": "Library Staff"
            },
            "additionalType": doc_type,
            "dateCreated": row.get("date"),
            "fileFormat": os.path.splitext(filename)[-1].replace('.', '').upper(),
            "description": f"{doc_type} document for {committee} dated {row.get('date')}.",
            "isBasedOn": row.get("original_name"),
            "license": "Open/Public per institutional policy",
            "checksum": {
                "algorithm": "SHA-256",
                "value": checksum
            }
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata written: {json_path}")

    # Summary
    print("\n--- Metadata Generation Summary ---")
    print(f"Created: {created}, Updated: {updated}, Skipped (unchanged): {skipped}")
    print("\nFile type counts:")
    for ext, count in sorted(file_type_counts.items()):
        print(f"  {ext.upper()}: {count}")


# Generate metadata for related documents directory because I had been skipping operations on it due to the very unique file names.
# Realizing this ought to also have fixity applied, this function was needed.
def run_related_documents(base_dir="data/Processed_Committees", skip_existing=False):
    print("\n--- Processing Related Documents ---")
    file_type_counts = defaultdict(int)
    created = updated = skipped = 0

    for committee in os.listdir(base_dir):
        committee_path = os.path.join(base_dir, committee)
        if not os.path.isdir(committee_path):
            continue

        related_path = os.path.join(committee_path, "Related Documents")
        if not os.path.isdir(related_path):
            continue

        for filename in os.listdir(related_path):
            if filename.endswith(".json"):
                continue
            full_path = os.path.join(related_path, filename)
            if not os.path.isfile(full_path):
                continue

            checksum = calculate_sha256(full_path)
            file_type = os.path.splitext(filename)[-1].lower()
            file_type_counts[file_type] += 1
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(related_path, base_name + ".json")

            # Track if we're creating a new file or updating an existing one
            if os.path.exists(json_path):
                updated += 1
            else:
                created += 1

            # Always generate new metadata and write to JSON
            metadata = {
                "@context": {
                    "@vocab": "http://schema.org/",
                    "checksum": "https://schema.org/checksum",
                    "algorithm": "https://schema.org/algorithm",
                    "value": "https://schema.org/value"
                },
                "@type": "CreativeWork",
                "name": filename,
                "creator": {
                    "@type": "Organization",
                    "name": "Library Staff"
                },
                "additionalType": "Related Document",
                "dateCreated": "unknown",
                "fileFormat": os.path.splitext(filename)[-1].replace('.', '').upper(),
                "description": f"Related document for {committee}.",
                "isBasedOn": filename,
                "license": "Open/Public per institutional policy",
                "checksum": {
                    "algorithm": "SHA-256",
                    "value": checksum
                }
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)

            print(f"Metadata written: {json_path}")

    print("\n--- Related Documents Summary ---")
    print(f"Created: {created}, Updated: {updated}, Skipped: {skipped}")
    print("File type counts:")
    for ext, count in sorted(file_type_counts.items()):
        print(f"  {ext.upper()}: {count}")