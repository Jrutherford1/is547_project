import os
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import spacy


# Load the spaCy model ONCE at module level - this is key for performance!
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])  # Only keep what you need

# Import the NLP functions from the existing module
from data_pipeline.nlp_term_extraction_preview import extract_text

# Import quality validation modules
from data_pipeline.text_quality import calculate_text_quality_score, is_garbage_text
from data_pipeline.entity_validation import (
    validate_entity, clean_entity_text, PERSON_FILTER_TERMS
)
from data_pipeline.nlp_quality_report import (
    QualityTracker, DocumentQualityRecord
)

def extract_entities_batch(
    file_paths: List[str],
    quality_tracker: Optional[QualityTracker] = None,
    text_quality_threshold: float = 0.35,
    validate_entities: bool = True
) -> List[Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, str]]], float]]:
    """
    Extracts entities from multiple files using batch processing with quality validation.

    Args:
        file_paths: List of file paths to process
        quality_tracker: Optional QualityTracker for recording metrics
        text_quality_threshold: Minimum text quality score to process (default 0.35)
        validate_entities: Whether to validate extracted entities (default True)

    Returns:
        List of tuples: (entities_dict, rejected_dict, quality_score)
        - entities_dict: {entity_type: [entity_strings]}
        - rejected_dict: {entity_type: [(entity, reason)]}
        - quality_score: float 0.0-1.0
    """
    # Extract texts from all files first, with quality scoring
    texts = []
    valid_paths = []
    quality_scores = []
    quality_results = []

    for file_path in file_paths:
        try:
            text = extract_text(file_path)
            if text and len(text.strip()) > 0:
                # Calculate text quality
                quality = calculate_text_quality_score(text)
                quality_scores.append(quality['overall_score'])
                quality_results.append(quality)

                if quality['is_valid'] and quality['overall_score'] >= text_quality_threshold:
                    texts.append(text)
                else:
                    # Text is garbage - use empty string but record the issue
                    texts.append("")
                    if quality_tracker:
                        quality_tracker.record_document(DocumentQualityRecord(
                            file_path=file_path,
                            text_quality_score=quality['overall_score'],
                            text_length=len(text),
                            extraction_success=False,
                            issues=quality['issues'],
                            entities_extracted={},
                            entities_rejected={}
                        ))
                valid_paths.append(file_path)
            else:
                texts.append("")
                valid_paths.append(file_path)
                quality_scores.append(0.0)
                quality_results.append({'overall_score': 0.0, 'is_valid': False, 'issues': ['empty_text']})

        except Exception as e:
            # Add empty text to maintain alignment
            texts.append("")
            valid_paths.append(file_path)
            quality_scores.append(0.0)
            quality_results.append({'overall_score': 0.0, 'is_valid': False, 'issues': [f'extraction_error:{str(e)[:50]}']})

    # Process all texts in a single batch - this is much faster!
    results = []
    for doc, file_path, quality_score, quality_result in zip(
        nlp.pipe(texts, batch_size=50), valid_paths, quality_scores, quality_results
    ):
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": []
        }
        rejected = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": []
        }

        # Extract entities by type
        for ent in doc.ents:
            if ent.label_ in entities:
                # Clean the entity text
                cleaned_text = clean_entity_text(ent.text)

                if not cleaned_text:
                    continue

                # Validate if enabled
                if validate_entities:
                    is_valid, reason = validate_entity(cleaned_text, ent.label_)

                    if is_valid:
                        if cleaned_text not in entities[ent.label_]:
                            entities[ent.label_].append(cleaned_text)
                            if quality_tracker:
                                quality_tracker.record_entity(ent.label_, cleaned_text, True, "")
                    else:
                        rejected[ent.label_].append((cleaned_text, reason))
                        if quality_tracker:
                            quality_tracker.record_entity(ent.label_, cleaned_text, False, reason)
                else:
                    # No validation - just add if not duplicate
                    if cleaned_text not in entities[ent.label_]:
                        entities[ent.label_].append(cleaned_text)

        # Sort each entity list
        for key in entities:
            entities[key] = sorted(entities[key])

        results.append((entities, rejected, quality_score))

    return results

def enhance_json_with_nlp(
    base_dir: str = "data/Processed_Committees",
    limit: Optional[int] = None,
    batch_size: int = 100,
    progress_interval: int = 10,
    skip_existing: bool = True,
    generate_report: bool = True,
    report_path: Optional[str] = None,
    text_quality_threshold: float = 0.35,
    validate_entities: bool = True
) -> Dict[str, Any]:
    """
    Enhances existing JSON-LD metadata files with NLP-extracted entities using batch processing.
    Includes quality validation and optional reporting.

    Args:
        base_dir: Base directory containing processed committee files
        limit: Optional limit on number of files to process
        batch_size: Number of files to process in each batch
        progress_interval: How often to print progress (in batches)
        skip_existing: Skip files that already have entities
        generate_report: Generate a quality report
        report_path: Path for quality report JSON (default: data/nlp_quality_report.json)
        text_quality_threshold: Minimum text quality score to process
        validate_entities: Whether to validate extracted entities

    Returns:
        Dict with processing statistics
    """
    import warnings
    import logging

    # Suppress PDF cropbox warnings and other pdfplumber warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)

    # Initialize quality tracker
    quality_tracker = QualityTracker() if generate_report else None
    if quality_tracker:
        quality_tracker.start()
    
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

                # Process the batch with quality validation
                try:
                    batch_results = extract_entities_batch(
                        doc_paths,
                        quality_tracker=quality_tracker,
                        text_quality_threshold=text_quality_threshold,
                        validate_entities=validate_entities
                    )

                    # Update JSON files with results
                    for (doc_path, json_path, committee), (entities, rejected, quality_score) in zip(files_to_process, batch_results):
                        try:
                            # Update counters (entities are already filtered by extract_entities_batch)
                            for entity_type, entity_list in entities.items():
                                for entity in entity_list:
                                    all_entities[entity_type][entity] += 1
                                    committee_entities[committee][entity_type][entity] += 1

                            # Record document quality if tracking
                            if quality_tracker:
                                entities_extracted = {k: len(v) for k, v in entities.items()}
                                entities_rejected = {k: len(v) for k, v in rejected.items()}
                                quality_tracker.record_document(DocumentQualityRecord(
                                    file_path=doc_path,
                                    text_quality_score=quality_score,
                                    text_length=0,  # Already processed
                                    extraction_success=True,
                                    issues=[],
                                    entities_extracted=entities_extracted,
                                    entities_rejected=entities_rejected
                                ))

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
            batch_results = extract_entities_batch(
                doc_paths,
                quality_tracker=quality_tracker,
                text_quality_threshold=text_quality_threshold,
                validate_entities=validate_entities
            )

            for (doc_path, json_path, committee), (entities, rejected, quality_score) in zip(files_to_process, batch_results):
                try:
                    # Update counters (entities are already filtered)
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            all_entities[entity_type][entity] += 1
                            committee_entities[committee][entity_type][entity] += 1

                    # Record document quality if tracking
                    if quality_tracker:
                        entities_extracted = {k: len(v) for k, v in entities.items()}
                        entities_rejected = {k: len(v) for k, v in rejected.items()}
                        quality_tracker.record_document(DocumentQualityRecord(
                            file_path=doc_path,
                            text_quality_score=quality_score,
                            text_length=0,
                            extraction_success=True,
                            issues=[],
                            entities_extracted=entities_extracted,
                            entities_rejected=entities_rejected
                        ))

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

    # Generate quality report if tracking
    if quality_tracker and generate_report:
        if report_path is None:
            report_path = "data/nlp_quality_report.json"
        quality_tracker.save_report(report_path)
        quality_tracker.print_summary()
        print(f"\nQuality report saved to: {report_path}")

    return {
        "processed": processed,
        "updated": updated,
        "already_enhanced": already_enhanced,
        "missing_files": missing_files,
        "skipped": skipped,
        "entities": dict(all_entities),
        "committee_entities": dict(committee_entities)
    }


def reprocess_all_entities(
    base_dir: str = "data/Processed_Committees",
    batch_size: int = 100,
    generate_report: bool = True,
    report_path: str = "data/nlp_quality_report.json",
    text_quality_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Reprocesses all documents with improved filtering.
    Clears existing entities before reprocessing.

    Args:
        base_dir: Base directory containing processed committee files
        batch_size: Number of files to process in each batch
        generate_report: Generate a quality report
        report_path: Path for quality report JSON
        text_quality_threshold: Minimum text quality score to process

    Returns:
        Dict with processing statistics
    """
    print("=" * 60)
    print("REPROCESSING ALL ENTITIES WITH IMPROVED VALIDATION")
    print("=" * 60)

    # Step 1: Clear existing entities from all JSON files
    print("\nStep 1: Clearing existing entities from JSON metadata files...")
    cleared = 0
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.json') and f != 'project_metadata.jsonld':
                json_path = os.path.join(root, f)
                try:
                    with open(json_path, 'r', encoding='utf-8') as file:
                        metadata = json.load(file)

                    # Clear entities and keywords
                    if 'entities' in metadata:
                        metadata['entities'] = {}
                    if 'keywords' in metadata:
                        metadata['keywords'] = []

                    with open(json_path, 'w', encoding='utf-8') as file:
                        json.dump(metadata, file, indent=4)
                    cleared += 1
                except Exception as e:
                    print(f"  Warning: Could not clear {json_path}: {e}")

    print(f"  Cleared entities from {cleared} JSON files")

    # Step 2: Run enhanced extraction
    print("\nStep 2: Running enhanced NLP extraction with quality validation...")
    results = enhance_json_with_nlp(
        base_dir=base_dir,
        limit=None,
        batch_size=batch_size,
        skip_existing=False,  # Force reprocess all
        generate_report=generate_report,
        report_path=report_path,
        text_quality_threshold=text_quality_threshold,
        validate_entities=True
    )

    print("\n" + "=" * 60)
    print("REPROCESSING COMPLETE")
    print("=" * 60)

    return results