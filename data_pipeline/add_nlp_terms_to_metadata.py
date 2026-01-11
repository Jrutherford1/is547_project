import os
import json
from collections import defaultdict, Counter
from pathlib import Path
import spacy


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
    Only prints summary information and suppresses PDF warnings.
    """
    import warnings
    import logging
    
    # Suppress PDF cropbox warnings and other pdfplumber warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)
    
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

    # First, count how many files we'll be processing
    total_json_files = 0
    for root, _, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]
        total_json_files += len(json_files)

    print(f"Starting NLP enhancement of {total_json_files} JSON metadata files...")
    if limit:
        print(f"Processing limit: {limit} files")

    # Collect files to process in batches
    files_to_process = []
    
    # Process files
    for root, _, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]

        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            base_name = Path(json_file).stem
            rel_path = os.path.relpath(json_path, base_dir)

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
                        
                        if limit and count >= limit:
                            break
                        continue
                except Exception:
                    # If there's any error reading the file, continue with processing
                    pass

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

            count += 1

            # Process in batches
            if len(files_to_process) >= batch_size or (limit and count >= limit):
                # Extract just the file paths for batch processing
                doc_paths = [item[0] for item in files_to_process]
                
                # Process the batch (suppress output)
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
                            
                            # Filter PERSON entities
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
                                
                        except Exception:
                            skipped += 1
                    
                    # Show progress every few batches
                    if processed % (batch_size * 5) == 0 and processed > 0:
                        print(f"Progress: {processed} files processed, {updated} updated")
                    
                except Exception:
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
                    
                    # Filter PERSON entities
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

                    # Add keywords from entities
                    keywords = []
                    if entities["PERSON"]:
                        keywords.extend(entities["PERSON"][:3])
                    if entities["ORG"]:
                        keywords.extend(entities["ORG"][:3])
                    if keywords:
                        metadata["keywords"] = keywords

                    # Save updated JSON-LD
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=4)

                    updated += 1
                    processed += 1
                        
                except Exception:
                    skipped += 1
                    
        except Exception:
            skipped += len(files_to_process)

    # Print final summary
    print(f"\n=== NLP Enhancement Complete ===")
    print(f"Files examined: {count}")
    print(f"Files processed: {processed}")
    print(f"JSON files updated: {updated}")
    print(f"Already enhanced: {already_enhanced}")
    print(f"Missing documents: {missing_files}")
    print(f"Skipped (errors): {skipped}")
    
    # Entity summary
    total_entities = sum(len(entities) for entities in all_entities.values())
    print(f"\nEntities extracted:")
    for entity_type, entities in all_entities.items():
        print(f"  {entity_type}: {len(entities)} unique")
    print(f"Total unique entities: {total_entities}")

    return {
        "processed": processed,
        "updated": updated,
        "already_enhanced": already_enhanced,
        "missing_files": missing_files,
        "skipped": skipped,
        "entities": dict(all_entities),
        "committee_entities": dict(committee_entities)
    }