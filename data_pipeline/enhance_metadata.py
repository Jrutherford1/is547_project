import os
import json
import hashlib
import pandas as pd
from collections import defaultdict


def calculate_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def enhance_all_metadata(
    csv_path="data/final_updated_committee_names.csv",
    base_dir="data/Processed_Committees",
    skip_existing=False
):
    """
    Enhanced metadata function that processes ALL files in one pass,
    both regular files and related documents, with a single summary at the end.
    
    Args
        csv_path: Path to the CSV file with file information
        base_dir: Base directory of processed files
        skip_existing: Whether to skip files that already have metadata
    """
    # Stats tracking
    file_type_counts = defaultdict(int)
    created = updated = skipped = missing = 0
    total_regular = 0
    related_count = 0
    committees_with_related = set()
    
    # Track which files we've processed
    processed_files = set()
    
    # Step 1: Process all files listed in the CSV
    try:
        df = pd.read_csv(csv_path)
        df.rename(columns={
            'Final File Name': 'final_filename',
            'Committee': 'committee',
            'Document Type': 'type',
            'Extracted Date': 'date',
            'Original File Name': 'original_name'
        }, inplace=True)
        
        for _, row in df.iterrows():
            committee = row['committee'].strip()
            doc_type = row['type'].strip()
            filename = row['final_filename'].strip()
            original_filename = row['original_name'].strip()
            
            # Generate file path
            rel_path = os.path.join(committee, doc_type, filename)
            full_path = os.path.join(base_dir, rel_path)
            
            # Skip if file doesn't exist
            if not os.path.isfile(full_path):
                missing += 1
                continue
                
            # Mark as processed (to avoid duplicate processing)
            processed_files.add(full_path)
            
            # Count stats
            if doc_type.lower() == "related documents":
                related_count += 1
                committees_with_related.add(committee)
            else:
                total_regular += 1
                
            # Get file info
            file_type = os.path.splitext(filename)[-1].lower()
            file_type_counts[file_type] += 1
            
            # Generate metadata JSON path
            base_name = os.path.splitext(filename)[0]
            folder_path = os.path.dirname(full_path)
            json_path = os.path.join(folder_path, base_name + ".json")
            
            # Check if metadata already exists
            if os.path.exists(json_path) and skip_existing:
                skipped += 1
                continue
            elif os.path.exists(json_path):
                updated += 1
            else:
                created += 1
            
            # Generate checksum
            checksum = calculate_sha256(full_path)
            
            # Create metadata
            metadata = {
                "@context": {
                    "@vocab": "http://schema.org/",
                    "checksum": "https://schema.org/checksum",
                    "algorithm": "https://schema.org/algorithm",
                    "value": "https://schema.org/value",
                    "originalFileName": "http://schema.org/alternateName"
                },
                "@type": "CreativeWork",
                "name": filename,
                "creator": {
                    "@type": "Organization",
                    "name": "Library Staff"
                },
                "additionalType": doc_type,
                "dateCreated": row.get("date", "unknown"),
                "fileFormat": os.path.splitext(filename)[-1].replace('.', '').upper(),
                "description": f"{doc_type} document for {committee}.",
                "isBasedOn": original_filename,
                "originalFileName": original_filename,
                "license": "Open/Public per institutional policy",
                "checksum": {
                    "algorithm": "SHA-256",
                    "value": checksum
                }
            }
            
            # Add date to description if available
            if row.get("date") and row.get("date") != "unknown":
                metadata["description"] = f"{doc_type} document for {committee} dated {row.get('date')}."
            
            # Write metadata
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
    
    except Exception as e:
        print(f"Error processing CSV files: {e}")
    
    # Step 2: Find any files not in the CSV (mostly in Related Documents folders)
    manually_added = 0
    for root, _, files in os.walk(base_dir):
        for file in files:
            # Skip JSON files and already processed files
            if file.endswith(".json"):
                continue
                
            full_path = os.path.join(root, file)
            if full_path in processed_files:
                continue
                
            # Skip if not a file
            if not os.path.isfile(full_path):
                continue
            
            # This is a file we haven't processed yet
            manually_added += 1
            
            # Get path components
            rel_path = os.path.relpath(full_path, base_dir)
            path_parts = rel_path.split(os.sep)
            
            # Need at least committee/folder/file structure
            if len(path_parts) < 2:
                continue
                
            committee = path_parts[0]
            folder = path_parts[1] if len(path_parts) > 2 else "Unknown"
            
            # Check if this is a related document
            if folder.lower() == "related documents":
                related_count += 1
                committees_with_related.add(committee)
            
            # Get file info
            file_type = os.path.splitext(file)[-1].lower()
            file_type_counts[file_type] += 1
            
            # Create JSON path
            base_name = os.path.splitext(file)[0]
            json_path = os.path.join(os.path.dirname(full_path), base_name + ".json")
            
            # Check if metadata already exists
            if os.path.exists(json_path) and skip_existing:
                skipped += 1
                continue
            elif os.path.exists(json_path):
                updated += 1
            else:
                created += 1
            
            # Generate checksum
            checksum = calculate_sha256(full_path)
            
            # Create metadata
            metadata = {
                "@context": {
                    "@vocab": "http://schema.org/",
                    "checksum": "https://schema.org/checksum",
                    "algorithm": "https://schema.org/algorithm",
                    "value": "https://schema.org/value",
                    "originalFileName": "http://schema.org/alternateName"
                },
                "@type": "CreativeWork",
                "name": file,
                "creator": {
                    "@type": "Organization",
                    "name": "Library Staff"
                },
                "additionalType": folder,
                "dateCreated": "unknown",
                "fileFormat": file_type.replace('.', '').upper(),
                "description": f"{folder} document for {committee}.",
                "isBasedOn": file,
                "originalFileName": file,
                "license": "Open/Public per institutional policy",
                "checksum": {
                    "algorithm": "SHA-256",
                    "value": checksum
                }
            }
            
            # Write metadata
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
    
    # Print single summary at the end
    print("=== Metadata Enhancement Complete ===")
    print(f"Total files processed: {total_regular + related_count + manually_added}")
    print(f"  - Regular files from CSV: {total_regular}")
    print(f"  - Related documents from CSV: {related_count}")
    print(f"  - Additional files found: {manually_added}")
    print(f"Committees with related documents: {len(committees_with_related)}")
    print(f"Created: {created}, Updated: {updated}, Skipped: {skipped}, Missing: {missing}")
    
    print("\nFile type counts:")
    for ext, count in sorted(file_type_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {ext.upper() or '(no extension)'}: {count}")


# For backward compatibility, these functions now just call the combined function
def enhance_metadata(csv_path="data/final_updated_committee_names.csv", 
                   base_dir="data/Processed_Committees", 
                   skip_existing=False):
    """Legacy function that calls the new combined function"""
    enhance_all_metadata(csv_path, base_dir, skip_existing)


def run_related_documents(base_dir="data/Processed_Committees", skip_existing=False):
    """Legacy function that calls the new combined function"""
    enhance_all_metadata(base_dir=base_dir, skip_existing=skip_existing)