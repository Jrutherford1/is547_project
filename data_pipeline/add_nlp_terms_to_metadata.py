import os
import json
from collections import defaultdict, Counter
from pathlib import Path
import spacy
import hashlib

# Load the spaCy model ONCE at module level - this is key for performance!
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])  # Only keep what you need

# Import the NLP functions from the existing module
from data_pipeline.nlp_term_extraction_preview import extract_text

# In your add_nlp_terms_to_metadata.py file
PERSON_FILTER_TERMS = {
    "Library", "Librarian", "Librarians", "Staff", "Committee", 
    "Chair", "Chairperson", "Member", "Members", "Director",
    "University", "UIUC", "Illinois", "Department"
}

def extract_entities_batch(file_paths):
    """
    Extracts entities from multiple files using batch processing.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        List of dictionaries with entity types as keys and lists of entities as values
    """
    # Extract texts from all files first
    texts = []
    valid_paths = []
    
    for file_path in file_paths:
        try:
            text = extract_text(file_path)
            if text and len(text.strip()) > 0:  # Only process non-empty texts
                texts.append(text)
                valid_paths.append(file_path)
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            # Add empty text to maintain alignment
            texts.append("")
            valid_paths.append(file_path)
    
    # Process all texts in a single batch - this is much faster!
    results = []
    for doc, file_path in zip(nlp.pipe(texts, batch_size=50), valid_paths):
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geopolitical entities (countries, cities, etc.)
            "DATE": []
        }
        
        # Extract entities by type
        for ent in doc.ents:
            if ent.label_ in entities:
                # Normalize and add only if not already present
                cleaned_text = ent.text.strip()
                if cleaned_text and cleaned_text not in entities[ent.label_]:
                    entities[ent.label_].append(cleaned_text)
        
        # Sort each entity list
        for key in entities:
            entities[key] = sorted(entities[key])
            
        results.append(entities)
    
    return results

def enhance_json_with_nlp(base_dir="data/Processed_Committees", limit=None, batch_size=100, progress_interval=10, skip_existing=True):
    """
    Enhances existing JSON-LD metadata files with NLP-extracted entities using batch processing.
    """
    count = 0
    processed = 0
    updated = 0
    skipped = 0
    already_enhanced = 0
    missing_files = 0
    supported_extensions = [".txt", ".docx", ".pptx", ".pdf"]

    # Track entity statistics
    all_entities = {
        "PERSON": Counter(),
        "ORG": Counter(),
        "GPE": Counter(),
        "DATE": Counter()
    }

    # Also track by committee
    committee_entities = defaultdict(lambda: {
        "PERSON": Counter(),
        "ORG": Counter(),
        "GPE": Counter(),
        "DATE": Counter()
    })

    print(f"Enhancing JSON-LD metadata with NLP entities using batch processing...")

    # First, count how many files we'll be processing
    total_json_files = 0
    for root, _, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]
        total_json_files += len(json_files)

    if limit:
        print(f"Will process up to {limit} of {total_json_files} JSON files")
    else:
        print(f"Found {total_json_files} JSON files to process")

    # Collect files to process in batches
    files_to_process = []
    json_paths = []
    
    # Process files
    for root, _, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]

        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            base_name = Path(json_file).stem
            rel_path = os.path.relpath(json_path, base_dir)

            # Show periodic progress updates
            if count % progress_interval == 0:
                limit_text = f"/{limit}" if limit else f"/{total_json_files}"
                print(f"Processing file {count}{limit_text}: {rel_path}")

            # Extract committee name from path
            path_parts = rel_path.split(os.sep)
            committee = path_parts[0] if len(path_parts) > 0 else "Unknown"

            # First, check if file already has entities
            if skip_existing:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # If entities are already present, skip this file
                    if "entities" in metadata and any(metadata["entities"].values()):
                        # Still update our counters with existing entities
                        if "entities" in metadata:
                            for entity_type, entity_list in metadata["entities"].items():
                                for entity in entity_list:
                                    if entity_type in all_entities:
                                        all_entities[entity_type][entity] += 1
                                        committee_entities[committee][entity_type][entity] += 1
                        
                        already_enhanced += 1
                        count += 1
                        
                        if count % progress_interval == 0:
                            print(f"  Skipped - already has entities")
                        
                        if limit and count >= limit:
                            break
                        continue
                except Exception as e:
                    # If there's any error reading the file, continue with processing
                    print(f"Error reading {json_path}: {e}")

            # Try to find corresponding document file
            doc_found = False
            for ext in supported_extensions:
                doc_path = os.path.join(root, base_name + ext)
                if os.path.exists(doc_path):
                    doc_found = True
                    files_to_process.append((doc_path, json_path, committee))
                    break

            if not doc_found:
                missing_files += 1
                if count % progress_interval == 0:
                    print(f"  No corresponding document found")

            count += 1

            # Process in batches
            if len(files_to_process) >= batch_size or (limit and count >= limit):
                # Extract just the file paths for batch processing
                doc_paths = [item[0] for item in files_to_process]
                
                # Process the batch
                try:
                    batch_results = extract_entities_batch(doc_paths)
                    
                    # Update JSON files with results
                    for (doc_path, json_path, committee), entities in zip(files_to_process, batch_results):
                        try:
                            # Update counters
                            for entity_type, entity_list in entities.items():
                                for entity in entity_list:
                                    all_entities[entity_type][entity] += 1
                                    committee_entities[committee][entity_type][entity] += 1
                            
                            # Then when processing PERSON entities:
                            if "PERSON" in entities:
                                filtered_people = [person for person in entities["PERSON"] 
                                                  if person not in PERSON_FILTER_TERMS and 
                                                  not any(filter_term.lower() in person.lower() 
                                                         for filter_term in PERSON_FILTER_TERMS)]
                                entities["PERSON"] = filtered_people

                            # Load existing JSON-LD
                            with open(json_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)

                            # Add entities to metadata
                            if "entities" not in metadata:
                                metadata["entities"] = {}

                            # Add each entity type
                            for entity_type, entity_list in entities.items():
                                if entity_list:  # Only add non-empty lists
                                    metadata["entities"][entity_type] = entity_list

                            # Add keywords from entities (top organizations and people)
                            keywords = []

                            # Add up to 3 people
                            if entities["PERSON"]:
                                keywords.extend(entities["PERSON"][:min(3, len(entities["PERSON"]))])

                            # Add up to 3 organizations
                            if entities["ORG"]:
                                keywords.extend(entities["ORG"][:min(3, len(entities["ORG"]))])

                            # Add keywords to metadata if we found any
                            if keywords:
                                metadata["keywords"] = keywords

                            # Save updated JSON-LD
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=4)

                            updated += 1
                            processed += 1
                                
                        except Exception as e:
                            print(f"Error updating {json_path}: {e}")
                            skipped += 1
                    
                    # Show batch progress
                    print(f"\n--- Batch completed: processed {len(files_to_process)} files ---")
                    print(f"Total files processed so far: {processed}")
                    print(f"JSON files updated: {updated}")
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    skipped += len(files_to_process)
                
                # Clear the batch
                files_to_process = []

            if limit and count >= limit:
                break

        if limit and count >= limit:
            break

    # Process any remaining files in the last batch
    if files_to_process:
        doc_paths = [item[0] for item in files_to_process]
        try:
            batch_results = extract_entities_batch(doc_paths)
            
            for (doc_path, json_path, committee), entities in zip(files_to_process, batch_results):
                try:
                    # Update counters
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            all_entities[entity_type][entity] += 1
                            committee_entities[committee][entity_type][entity] += 1
                    
                    # Then when processing PERSON entities:
                    if "PERSON" in entities:
                        filtered_people = [person for person in entities["PERSON"] 
                                          if person not in PERSON_FILTER_TERMS and 
                                          not any(filter_term.lower() in person.lower() 
                                                 for filter_term in PERSON_FILTER_TERMS)]
                        entities["PERSON"] = filtered_people

                    # Load and update JSON
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    if "entities" not in metadata:
                        metadata["entities"] = {}

                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            metadata["entities"][entity_type] = entity_list

                    # Add keywords
                    keywords = []
                    if entities["PERSON"]:
                        keywords.extend(entities["PERSON"][:3])
                    if entities["ORG"]:
                        keywords.extend(entities["ORG"][:3])
                    if keywords:
                        metadata["keywords"] = keywords

                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=4)

                    updated += 1
                    processed += 1
                        
                except Exception as e:
                    print(f"Error updating {json_path}: {e}")
                    skipped += 1
        except Exception as e:
            print(f"Error processing final batch: {e}")

    # Update project metadata with top entities
    project_metadata_path = os.path.join(base_dir, "..", "project_metadata.jsonld")
    if os.path.exists(project_metadata_path):
        try:
            with open(project_metadata_path, 'r', encoding='utf-8') as f:
                project_metadata = json.load(f)

            # Add top entities to project metadata
            project_metadata["entities"] = {
                "topPeople": [person for person, _ in all_entities["PERSON"].most_common(20)],
                "topOrganizations": [org for org, _ in all_entities["ORG"].most_common(20)],
                "topLocations": [loc for loc, _ in all_entities["GPE"].most_common(10)]
            }

            # Update keywords in project metadata with top entities
            if "keywords" not in project_metadata:
                project_metadata["keywords"] = []

            # Add top 5 organizations to keywords if not already there
            for org, _ in all_entities["ORG"].most_common(5):
                if org not in project_metadata["keywords"]:
                    project_metadata["keywords"].append(org)

            # Save updated project metadata
            with open(project_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(project_metadata, f, indent=4)

            print(f"Updated project metadata with top entities")
        except Exception as e:
            print(f"Error updating project metadata: {e}")

    # Print summary
    print("\n=== NLP Entity Extraction Summary ===")
    print(f"Files processed: {processed}")
    print(f"JSON files updated: {updated}")
    print(f"Files already enhanced (skipped): {already_enhanced}")
    print(f"Files skipped due to errors: {skipped}")
    print(f"JSON files without corresponding documents: {missing_files}")

    print("\nTop 10 people mentioned:")
    for person, count in all_entities["PERSON"].most_common(10):
        print(f"  {person}: {count} mentions")

    print("\nTop 10 organizations mentioned:")
    for org, count in all_entities["ORG"].most_common(10):
        print(f"  {org}: {count} mentions")

    print("\nTop 5 locations mentioned:")
    for loc, count in all_entities["GPE"].most_common(5):
        print(f"  {loc}: {count} mentions")

    # Return statistics for further analysis if needed
    return {
        "processed": processed,
        "updated": updated,
        "skipped": skipped,
        "already_enhanced": already_enhanced,
        "missing": missing_files,
        "all_entities": all_entities,
        "committee_entities": committee_entities
    }

# If running this file directly, you can test the function
if __name__ == "__main__":
    # Test with a small batch first
    results = enhance_json_with_nlp(
        base_dir="data/Processed_Committees",
        limit=50,  # Start small to test performance
        batch_size=25,
        skip_existing=True
    )
    print("Test completed successfully!")