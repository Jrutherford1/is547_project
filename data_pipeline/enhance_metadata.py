# enhance_metadata.py
import os
import json
import hashlib
import pandas as pd
from collections import defaultdict

# Calculate SHA-256 checksum
def calculate_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# Generate JSON-LD metadata from main committee CSV
def enhance_metadata(
    csv_path="/mnt/data/manually_updated_committee_names.csv",
    base_dir="data/Processed_Committees",
    skip_existing=True
):
    df = pd.read_csv(csv_path)
    df.rename(columns={
        'Proposed File Name': 'final_filename',
        'Committee': 'committee',
        'Document Type': 'type',
        'Extracted Date': 'date',
        'Original File Name': 'original_name'
    }, inplace=True)
    df['final_filename'] = df['final_filename'].str.strip()

    file_type_counts = defaultdict(int)
    skipped = updated = created = 0

    for _, row in df.iterrows():
        committee = row['committee'].strip()
        doc_type = row['type'].strip()
        filename = row['final_filename'].strip()

        rel_path = os.path.join(committee, doc_type, filename)
        full_path = os.path.join(base_dir, rel_path)

        if not os.path.isfile(full_path):
            print(f"Missing: {rel_path}")
            continue

        file_type = os.path.splitext(filename)[-1].lower()
        file_type_counts[file_type] += 1
        checksum = calculate_sha256(full_path)
        base_name = os.path.splitext(filename)[0]
        folder_path = os.path.dirname(full_path)
        json_path = os.path.join(folder_path, base_name + ".json")

        if skip_existing and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    existing = json.load(f)
                    if existing.get("checksum", {}).get("value") == checksum:
                        skipped += 1
                        continue
                    else:
                        print(f"Checksum mismatch: {filename}, updating JSON-LD.")
                        updated += 1
                except Exception:
                    print(f"Corrupt JSON-LD for {filename}, regenerating.")
        else:
            created += 1

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

    print("\n--- JSON-LD Metadata Generation Summary ---")
    print(f"Created: {created}, Updated: {updated}, Skipped (unchanged): {skipped}")
    print("\nFile type counts:")
    for ext, count in sorted(file_type_counts.items()):
        print(f"  {ext.upper()}: {count}")


# Generate JSON-LD metadata for "Related Documents"
def run_related_documents(base_dir="data/Processed_Committees", skip_existing=True):
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

            if skip_existing and os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        existing = json.load(f)
                        if existing.get("checksum", {}).get("value") == checksum:
                            skipped += 1
                            continue
                        else:
                            print(f"Checksum mismatch: {filename}, updating JSON-LD.")
                            updated += 1
                    except Exception:
                        print(f"Unreadable JSON for {filename}, regenerating.")
            else:
                created += 1

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