import os
import json
from collections import Counter


def check_person_entities(base_dir="data/Processed_Committees", limit=10):
    """Check what PERSON entities are currently in the metadata files"""
    person_counter = Counter()
    files_checked = 0

    for root, _, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]

        for json_file in json_files:
            if files_checked >= limit:
                break

            json_path = os.path.join(root, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if "entities" in metadata and "PERSON" in metadata["entities"]:
                    for person in metadata["entities"]["PERSON"]:
                        person_counter[person] += 1

                files_checked += 1

            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    print(f"Checked {files_checked} files")
    print(f"Current PERSON entities found:")
    for person, count in person_counter.most_common(15):
        print(f"  {person}: {count} mentions")

    return person_counter


# Check what's currently in your metadata
current_entities = check_person_entities(limit=20)